# Seg-Weather-Robustness

基于 CARLA 的多天气语义分割小型研究项目，聚焦于**固定 DeepLabV3-ResNet50 Backbone + 天气感知训练策略**。仓库同时支持两条基线与天气 aware 课程学习方案，便于快速对比鲁棒性。

## 快速上手

```bash
# 准备环境（示例）
conda create -n seg-weather python=3.10 -y
conda activate seg-weather
pip install torch torchvision pyyaml matplotlib tensorboard
```

## 数据组织与划分

1. 启动 CARLA 服务器后，可运行 `python scripts/collect_multi_weather.py --samples-per-weather 200` 自动采集六种天气（脚本会在 `data/carla/<weather>` 下保存 RGB + 语义 mask）。更多细节见 `scripts/generate_carla_data.md`。
2. 运行 `python scripts/prepare_splits.py --data-root ./data --output ./data/splits` 生成 `train/val/test` 列表（行格式：`img mask weather`）。
3. 若已有自己的划分文件，将路径写入 YAML 配置中的 `train_list`、`val_list` 即可。

## 训练命令

- Baseline 0（仅晴天）：
  ```bash
  python train/train_baseline.py --cfg configs/baseline_clear.yaml
  ```
- Baseline 1（混合训练）：
  ```bash
  python train/train_baseline.py --cfg configs/baseline_mixed.yaml
  ```
- Ours（课程 + 天气增强 + 加权采样）：
  ```bash
  python train/train_weather_aware.py --cfg configs/ours_weather_aware.yaml
  ```

训练脚本会在 `logs/checkpoints/` 下保存最佳 mIoU 的权重，可在配置中通过 `save_path` 自定义。

## TensorBoard 可视化

- 默认会把训练 loss、验证 mIoU、学习率等指标写入 `logs/tensorboard/<experiment_name>`。
- 可在配置里通过 `tensorboard_log_dir` 更改输出路径。
- 使用 `tensorboard --logdir logs/tensorboard`（或你自定义的目录）即可实时查看训练与 evaluation 曲线。

## 评估与可视化

- 整体 mIoU：使用 `eval/eval_segmentation.py` 中的 `evaluate`，或在训练脚本中直接复用。
- 分天气 mIoU：调用 `evaluate_per_weather`，DataLoader 需返回 `batch["weather"]` 字段。
- 预测可视化：`utils/visualization.py` 提供 `save_visual_comparison`，可生成「RGB / 多模型预测 / GT」拼图。

## 目录结构

```
configs/                # YAML 配置
datasets/               # CARLA 数据集定义
models/                 # DeepLabV3 封装
samplers/               # 天气加权采样器
train/                  # 训练脚本（基线 & 天气 aware）
eval/                   # mIoU 评估工具
utils/                  # 通用变换、指标、runner、可视化
scripts/                # 数据生成说明 + 划分脚本
data/                   # 占位目录，放置CARLA数据
logs/                   # 日志 & checkpoint 输出
```

## 备注

- 默认 `image_size` 为 `[512, 256]`（H, W），可在配置中修改。
- Mask 需使用语义类别索引（0 ~ num_classes-1），背景/忽略值默认 255。
- 可通过在 YAML 中添加 `curriculum` 字段自定义阶段式训练（详见 `train/train_weather_aware.py`）。
