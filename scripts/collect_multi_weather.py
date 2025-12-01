#!/usr/bin/env python3
"""CARLA 多天气数据采集脚本。

示例：
    python scripts/collect_multi_weather.py --samples-per-weather 200 --output-root ./data/carla

脚本会在 data/carla/<weather_name>/images 与 masks 目录下保存 RGB 与语义分割 PNG。
"""

from __future__ import annotations

import argparse
import glob
import logging
import queue
import random
import sys
from math import isclose
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import carla
except ImportError:
    # 兼容从 CARLA 发行包运行：自动搜索 egg
    CARLA_EGG = sorted(Path(__file__).resolve().parents[2].glob("**/carla-*py3*.egg"))
    if CARLA_EGG:
        sys.path.append(str(CARLA_EGG[-1]))
        import carla  # type: ignore  # noqa: E402
    else:  # pragma: no cover
        raise

LOGGER = logging.getLogger("collect_multi_weather")


def _build_weather_presets() -> Dict[str, carla.WeatherParameters]:
    """
    返回 {目录名: WeatherParameters}，覆盖 Clear/Rain/Fog 在
    日间（Noon）、黄昏（Sunset）、夜间（Night）的组合。
    目录名直接采用官方 WeatherParameters 名称，避免歧义。
    """

    candidate_names = [
        # Clear
        "ClearNoon",
        "ClearSunset",
        "ClearNight",
        # Light rain
        "SoftRainNoon",
        "SoftRainSunset",
        "SoftRainNight",
        # Heavy rain
        "HardRainNoon",
        "HardRainSunset",
        "HardRainNight",
        # Fog
        # "FoggyNoon",
        # "FoggySunset",
        # "FoggyNight",
    ]

    presets: Dict[str, carla.WeatherParameters] = {}
    for name in candidate_names:
        weather = getattr(carla.WeatherParameters, name, None)
        if weather is None:
            LOGGER.warning("当前 CARLA 版本缺少 %s 预设，将跳过该天气", name)
            continue
        presets[name] = weather

    if not presets:
        raise RuntimeError("未能构建任何天气预设，请检查 CARLA 版本是否完整")
    return presets


WEATHER_PRESETS: Dict[str, carla.WeatherParameters] = _build_weather_presets()


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA 多天气采集脚本")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None, help="可选：指定地图名，如 Town10HD")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--samples-per-weather", type=int, default=200)
    parser.add_argument("--camera-height", type=float, default=1.6)
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--output-root", default="./data/carla")
    parser.add_argument(
        "--weathers",
        nargs="*",
        default=list(WEATHER_PRESETS.keys()),
        help="可选：只采集部分天气，使用WeatherParameters名称",
    )
    parser.add_argument("--samples-interval", type=int, default=5, help="隔多少个 tick 记录一次样本")
    parser.add_argument("--vehicle-filter", default="vehicle.audi.tt", help="spawn车型过滤器")
    parser.add_argument("--warmup-ticks", type=int, default=60, help="切换天气后用于稳定的额外ticks")
    parser.add_argument("--no-autopilot", action="store_true", help="关闭自动驾驶（需要自己控制车辆）")
    return parser.parse_args()


def _make_output_dirs(root: Path, weather: str) -> Tuple[Path, Path]:
    img_dir = root / weather / "images"
    mask_dir = root / weather / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, mask_dir


def _spawn_vehicle(world: carla.World, blueprint_filter: str, seed: int) -> carla.Actor:
    random.seed(seed)
    blueprints = world.get_blueprint_library().filter(blueprint_filter)
    blueprint = random.choice(blueprints)
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("当前地图没有可用 spawn points")
    transform = random.choice(spawn_points)
    vehicle = world.spawn_actor(blueprint, transform)
    return vehicle


def _spawn_camera(world: carla.World, parent: carla.Actor, args) -> Tuple[carla.Sensor, carla.Sensor]:
    bp_lib = world.get_blueprint_library()
    rgb_bp = bp_lib.find("sensor.camera.rgb")
    seg_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    for bp in [rgb_bp, seg_bp]:
        bp.set_attribute("image_size_x", str(args.image_width))
        bp.set_attribute("image_size_y", str(args.image_height))
        bp.set_attribute("fov", str(args.camera_fov))
    camera_transform = carla.Transform(
        carla.Location(x=0.8, z=args.camera_height),
        carla.Rotation(pitch=-5.0),
    )
    rgb_cam = world.spawn_actor(rgb_bp, camera_transform, attach_to=parent)
    seg_cam = world.spawn_actor(seg_bp, camera_transform, attach_to=parent)
    return rgb_cam, seg_cam


def _enable_synchronous(world: carla.World):
    settings = world.get_settings()
    if not settings.synchronous_mode:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / 20.0
        world.apply_settings(settings)
    return settings


def _restore_world(world: carla.World, original_settings: carla.WorldSettings):
    world.apply_settings(original_settings)


def _listen(sensor: carla.Sensor, q: queue.Queue):
    sensor.listen(q.put)


def _drain_queue(q: "queue.Queue[carla.SensorData]"):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        return


def _weather_close(a: carla.WeatherParameters, b: carla.WeatherParameters, tol: float = 1.0) -> bool:
    """判断两个天气参数是否足够接近。"""

    keys = [
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wind_intensity",
        "sun_altitude_angle",
        "fog_density",
        "fog_distance",
        "wetness",
    ]
    return all(isclose(getattr(a, k), getattr(b, k), abs_tol=tol) for k in keys)


def _apply_weather(
    world: carla.World,
    weather_name: str,
    weather_params: carla.WeatherParameters,
    warmup_ticks: int,
    max_retries: int = 5,
):
    """
    确保天气真正生效；若连续多次失败则抛出异常而不是写入错误目录。
    """

    for attempt in range(1, max_retries + 1):
        world.set_weather(weather_params)
        for _ in range(max(warmup_ticks, 0)):
            world.tick()
        current = world.get_weather()
        if _weather_close(current, weather_params):
            LOGGER.info(
                "天气 %s 已应用 (cloudiness=%.1f, rain=%.1f, sun_alt=%.1f)",
                weather_name,
                current.cloudiness,
                current.precipitation,
                current.sun_altitude_angle,
            )
            return
        LOGGER.warning(
            "天气 %s 未成功应用（尝试 %d/%d），当前 sun_alt=%.1f, rain=%.1f",
            weather_name,
            attempt,
            max_retries,
            current.sun_altitude_angle,
            current.precipitation,
        )
    raise RuntimeError(f"连续 {max_retries} 次无法设置天气 {weather_name}，请检查是否有其他脚本在修改天气")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if args.town:
        world = client.load_world(args.town)

    original_settings = _enable_synchronous(world)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    vehicle = None
    sensors: List[carla.Actor] = []

    try:
        vehicle = _spawn_vehicle(world, args.vehicle_filter, args.seed)
        if not args.no_autopilot:
            vehicle.set_autopilot(True, traffic_manager.get_port())
        rgb_cam, seg_cam = _spawn_camera(world, vehicle, args)
        sensors.extend([rgb_cam, seg_cam])

        rgb_queue: "queue.Queue[carla.Image]" = queue.Queue()
        seg_queue: "queue.Queue[carla.Image]" = queue.Queue()
        _listen(rgb_cam, rgb_queue)
        _listen(seg_cam, seg_queue)

        output_root = Path(args.output_root).resolve()
        rng = random.Random(args.seed)
        world.tick()  # prime sensors

        for weather_name in args.weathers:
            if weather_name not in WEATHER_PRESETS:
                LOGGER.warning("跳过未知天气 %s", weather_name)
                continue
            _apply_weather(world, weather_name, WEATHER_PRESETS[weather_name], args.warmup_ticks)
            _drain_queue(rgb_queue)
            _drain_queue(seg_queue)
            img_dir, mask_dir = _make_output_dirs(output_root, weather_name)
            LOGGER.info("开始采集 %s，目标样本数=%d", weather_name, args.samples_per_weather)
            captured = 0
            frame_skip = 0
            while captured < args.samples_per_weather:
                world.tick()
                frame_skip = (frame_skip + 1) % max(args.samples_interval, 1)
                if frame_skip != 0:
                    continue
                try:
                    rgb_image = rgb_queue.get(timeout=5.0)
                    seg_image = seg_queue.get(timeout=5.0)
                except queue.Empty:
                    LOGGER.warning("等待传感器数据超时，重试...")
                    continue
                frame_id = rgb_image.frame
                rgb_path = img_dir / f"{weather_name}_{frame_id:06d}.png"
                mask_path = mask_dir / f"{weather_name}_{frame_id:06d}.png"
                rgb_image.save_to_disk(str(rgb_path))
                seg_image.save_to_disk(str(mask_path), carla.ColorConverter.CityScapesPalette)
                captured += 1
                if captured % 20 == 0:
                    LOGGER.info("%s 已采集 %d/%d", weather_name, captured, args.samples_per_weather)
            LOGGER.info("%s 完成", weather_name)

    finally:
        LOGGER.info("清理并恢复世界设置...")
        for sensor in sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except RuntimeError:
                pass
        if vehicle is not None:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except RuntimeError:
                pass
        traffic_manager.set_synchronous_mode(False)
        _restore_world(world, original_settings)


if __name__ == "__main__":
    main()
