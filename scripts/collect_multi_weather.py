#!/usr/bin/env python3
"""CARLA multi-weather data collection script.

Example:
    python scripts/collect_multi_weather.py --samples-per-weather 200 --output-root ./data/carla

The script saves RGB and semantic segmentation PNG files in data/carla/<weather_name>/images and masks directories.
"""

from __future__ import annotations

import argparse
import logging
import queue
import random
import sys
from math import isclose
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import carla
except ImportError:
    # Compatible with running from CARLA distribution: auto-search for egg
    CARLA_EGG = sorted(Path(__file__).resolve().parents[2].glob("**/carla-*py3*.egg"))
    if CARLA_EGG:
        sys.path.append(str(CARLA_EGG[-1]))
        import carla  # type: ignore  # noqa: E402
    else:  # pragma: no cover
        raise

LOGGER = logging.getLogger("collect_multi_weather")


def _build_weather_presets() -> Dict[str, carla.WeatherParameters]:
    """
    Returns {directory_name: WeatherParameters}, covering Clear/Rain combinations
    in daytime (Noon), sunset (Sunset), and nighttime (Night).
    Directory names directly use official WeatherParameters names to avoid ambiguity.
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
        # Fog (uncomment if needed)
        # "FoggyNoon",
        # "FoggySunset",
        # "FoggyNight",
    ]

    presets: Dict[str, carla.WeatherParameters] = {}
    for name in candidate_names:
        weather = getattr(carla.WeatherParameters, name, None)
        if weather is None:
            LOGGER.warning("Current CARLA version missing %s preset, skipping this weather", name)
            continue
        presets[name] = weather

    if not presets:
        raise RuntimeError("Failed to build any weather presets, please check if CARLA version is complete")
    return presets


WEATHER_PRESETS: Dict[str, carla.WeatherParameters] = _build_weather_presets()


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA multi-weather collection script")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None, help="Optional: specify map name, e.g., Town10HD")
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
        help="Optional: collect only specific weathers, use WeatherParameters names",
    )
    parser.add_argument("--samples-interval", type=int, default=50, help="Record sample every N ticks")
    parser.add_argument("--vehicle-filter", default="vehicle.audi.tt", help="Vehicle blueprint filter for spawning")
    parser.add_argument("--warmup-ticks", type=int, default=60, help="Extra ticks for stabilization after weather change")
    parser.add_argument(
        "--respawn-warmup-ticks",
        type=int,
        default=40,
        help="Extra ticks for trajectory stabilization after respawning vehicle (avoid saving frames before vehicle stabilizes)",
    )
    parser.add_argument("--no-autopilot", action="store_true", help="Disable autopilot (requires manual vehicle control)")
    parser.add_argument(
        "--save-color-mask",
        action="store_true",
        help="Also save CityScapesPalette color mask for visualization (training still uses Raw labels)",
    )
    return parser.parse_args()


def _make_output_dirs(root: Path, weather: str) -> Tuple[Path, Path]:
    img_dir = root / weather / "images"
    mask_dir = root / weather / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, mask_dir


def _spawn_vehicle(
    world: carla.World,
    blueprint_filter: str,
    rng: random.Random,
    spawn_transform: carla.Transform | None = None,
) -> carla.Actor:
    """Spawn a vehicle at specified or random spawn point (uses unified RNG to avoid resetting random seed each time)."""
    blueprints = world.get_blueprint_library().filter(blueprint_filter)
    if not blueprints:
        raise RuntimeError(f"No vehicle blueprint found matching {blueprint_filter}")
    blueprint = rng.choice(blueprints)

    if spawn_transform is None:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Current map has no available spawn points")
        spawn_transform = rng.choice(spawn_points)

    vehicle = world.spawn_actor(blueprint, spawn_transform)
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
        settings.fixed_delta_seconds = 1 / 20.0  # 20 Hz
        world.apply_settings(settings)
    return settings


def _restore_world(world: carla.World, original_settings: carla.WorldSettings):
    world.apply_settings(original_settings)


def _listen(sensor: carla.Sensor, q: queue.Queue):
    """Listen to sensor output, drop old frames when queue is full to prevent memory overflow."""

    def callback(data):
        try:
            q.put_nowait(data)
        except queue.Full:
            # Drop this frame if queue is full, maintain maxsize limit
            pass

    sensor.listen(callback)


def _drain_queue(q: "queue.Queue[carla.SensorData]"):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        return


def _weather_close(a: carla.WeatherParameters, b: carla.WeatherParameters, tol: float = 1.0) -> bool:
    """Check if two weather parameters are close enough."""

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
    Ensure weather is actually applied; raise exception after multiple consecutive failures
    instead of writing to wrong directory.
    """

    for attempt in range(1, max_retries + 1):
        world.set_weather(weather_params)
        for _ in range(max(warmup_ticks, 0)):
            world.tick()
        current = world.get_weather()
        if _weather_close(current, weather_params):
            LOGGER.info(
                "Weather %s applied (cloudiness=%.1f, rain=%.1f, sun_alt=%.1f)",
                weather_name,
                current.cloudiness,
                current.precipitation,
                current.sun_altitude_angle,
            )
            return
        LOGGER.warning(
            "Weather %s not successfully applied (attempt %d/%d), current sun_alt=%.1f, rain=%.1f",
            weather_name,
            attempt,
            max_retries,
            current.sun_altitude_angle,
            current.precipitation,
        )
    raise RuntimeError(f"Failed to set weather {weather_name} after {max_retries} attempts, check if other scripts are modifying weather")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    client = carla.Client(args.host, args.port)
    # Give tick more time to avoid timeout on large maps / first tick
    client.set_timeout(120.0)
    world = client.get_world()
    if args.town:
        world = client.load_world(args.town)

    original_settings = _enable_synchronous(world)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    vehicle: carla.Actor | None = None
    sensors: List[carla.Actor] = []

    # Unified random number & spawn points (ensures reproducibility while avoiding same position every time)
    rng = random.Random(args.seed)
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Current map has no available spawn points")

    output_root = Path(args.output_root).resolve()

    try:
        rgb_cam: carla.Sensor | None = None
        seg_cam: carla.Sensor | None = None
        # Limit queue size to prevent memory overflow from frame accumulation
        rgb_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=1)
        seg_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=1)

        # Pre-tick to activate sensors
        world.tick()

        for weather_name in args.weathers:
            if weather_name not in WEATHER_PRESETS:
                LOGGER.warning("Skipping unknown weather %s", weather_name)
                continue

            # Randomly select a spawn point before each weather
            spawn_tf = rng.choice(spawn_points)

            if vehicle is None:
                # First time: actually spawn a vehicle
                vehicle = _spawn_vehicle(world, args.vehicle_filter, rng, spawn_tf)
                if not args.no_autopilot:
                    vehicle.set_autopilot(True, traffic_manager.get_port())
                rgb_cam, seg_cam = _spawn_camera(world, vehicle, args)
                sensors.extend([rgb_cam, seg_cam])
                _listen(rgb_cam, rgb_queue)
                _listen(seg_cam, seg_queue)
            else:
                # Subsequent weathers: directly teleport vehicle to new spawn point
                vehicle.set_transform(spawn_tf)

            # Switch weather and warmup to ensure image/lighting stability
            _apply_weather(world, weather_name, WEATHER_PRESETS[weather_name], args.warmup_ticks)
            _drain_queue(rgb_queue)
            _drain_queue(seg_queue)

            img_dir, mask_dir = _make_output_dirs(output_root, weather_name)
            LOGGER.info("Starting collection for %s, target samples=%d", weather_name, args.samples_per_weather)

            captured = 0
            frame_skip = 0
            segment_size = 3  # Change spawn point every 3 images

            while captured < args.samples_per_weather:
                world.tick()
                frame_skip = (frame_skip + 1) % max(args.samples_interval, 1)
                if frame_skip != 0:
                    continue

                try:
                    rgb_image = rgb_queue.get(timeout=5.0)
                    seg_image = seg_queue.get(timeout=5.0)
                except queue.Empty:
                    LOGGER.warning("Timeout waiting for sensor data, retrying...")
                    continue

                frame_id = rgb_image.frame
                rgb_path = img_dir / f"{weather_name}_{frame_id:06d}.png"
                mask_path = mask_dir / f"{weather_name}_{frame_id:06d}.png"
                rgb_image.save_to_disk(str(rgb_path))
                # Save Raw (trainId) semantic labels for training
                seg_image.save_to_disk(str(mask_path), carla.ColorConverter.Raw)
                # Optional: also save a color visualization (CityScapesPalette)
                if args.save_color_mask:
                    vis_path = mask_dir / f"{weather_name}_{frame_id:06d}_color.png"
                    seg_image.save_to_disk(str(vis_path), carla.ColorConverter.CityScapesPalette)

                captured += 1
                if captured % 20 == 0:
                    LOGGER.info("%s collected %d/%d", weather_name, captured, args.samples_per_weather)

                # Change spawn point every segment_size samples to further increase scene diversity
                if captured % segment_size == 0 and captured < args.samples_per_weather:
                    new_tf = rng.choice(spawn_points)
                    LOGGER.info("%s: collected %d samples, switching to new spawn point", weather_name, captured)
                    vehicle.set_transform(new_tf)
                    # After position refresh, wait a bit before continuing sampling to avoid vehicle/trajectory not yet stable
                    for _ in range(max(args.respawn_warmup_ticks, 0)):
                        world.tick()
                    _drain_queue(rgb_queue)
                    _drain_queue(seg_queue)

            LOGGER.info("%s completed", weather_name)

    finally:
        LOGGER.info("Cleaning up and restoring world settings...")
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
        # These two steps may also fail after timeout, so wrap in defensive layer
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            LOGGER.warning("traffic_manager cleanup failed (server may have disconnected), ignoring.")
        try:
            _restore_world(world, original_settings)
        except RuntimeError:
            LOGGER.warning("world settings restoration failed (server may have disconnected), ignoring.")


if __name__ == "__main__":
    main()