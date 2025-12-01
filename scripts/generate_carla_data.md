# CARLA 数据导出指南

本项目假设使用 CARLA 自带的 Python API 采集多天气下的 RGB 与语义 mask。以下流程仅供参考，可根据自己的采集脚本调整。

1. 启动 CARLA 服务器（例如 `./CarlaUE4.sh -quality-level=Epic`）。
2. 运行官方 `PythonAPI/examples/generate_traffic.py` 或自定义采集脚本，针对 **晴/雨/雾 × 日间/黄昏/夜间** 的组合采集（`scripts/collect_multi_weather.py` 默认遍历下列 `WeatherParameters` 名称，并以该名称创建文件夹）：
   - 晴天：`ClearNoon`, `ClearSunset`, `ClearNight`
   - 小雨：`SoftRainNoon`, `SoftRainSunset`, `SoftRainNight`
   - 大雨：`HardRainNoon`, `HardRainSunset`, `HardRainNight`
   - 大雾：`FoggyNoon`, `FoggySunset`, `FoggyNight`（如当前 CARLA 版本缺少某个 Foggy 预设会自动跳过）
3. 脚本需同时保存：
   - 原始 RGB 图像，命名为 `data/carla/<weather>/images/*.png`
   - 语义标签（`CityScapesPalette`），保存为 `data/carla/<weather>/masks/*.png`
4. 推荐将相同帧号的 RGB 与 mask 文件名保持一致，方便 `CarlaWeatherDataset` 自动匹配。
5. 数据采集完成后，执行 `python scripts/prepare_splits.py --data-root ./data --output ./data/splits` 生成 train/val/test 列表。

> 如果已有 CARLA 导出工具，可在保存完成后创建软链接到 `data/carla/<weather>` 亦可。
