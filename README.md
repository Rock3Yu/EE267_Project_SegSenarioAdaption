# Seg-Weather-Robustness

A small research project on multi-weather semantic segmentation based on CARLA, focusing on **fixed DeepLabV3-ResNet50 backbone + weather-aware training strategies**. The repository supports both baseline methods and weather-aware curriculum learning approaches for quick robustness comparison.

## Quick Start

```bash
# Setup environment (example)
conda create -n seg-weather python=3.10 -y
conda activate seg-weather
pip install torch torchvision pyyaml matplotlib tensorboard
```

## Data Organization and Splitting

1. After starting the CARLA server, run `python scripts/collect_multi_weather.py --samples-per-weather 200` to automatically collect data from 9 weather conditions (the script saves RGB + semantic masks in `data/carla/<weather>`). The script supports:
   - Clear weather: ClearNoon, ClearSunset, ClearNight
   - Soft rain: SoftRainNoon, SoftRainSunset, SoftRainNight
   - Hard rain: HardRainNoon, HardRainSunset, HardRainNight

2. Run `python scripts/prepare_splits.py --data-root ./data --output ./data/splits` to generate `train/val/test` list files (format: `img_path mask_path weather`).

3. If you already have your own split files, specify their paths in the YAML config using `train_list` and `val_list`.

## Training Commands

- **Baseline 0** (Clear weather only):
  ```bash
  python train/train_baseline.py --cfg configs/baseline_clear.yaml
  ```

- **Baseline 1** (Mixed training):
  ```bash
  python train/train_baseline.py --cfg configs/baseline_mixed.yaml
  ```

- **Ours** (Curriculum learning + weather augmentation + weighted sampling):
  ```bash
  python train/train_weather_aware.py --cfg configs/ours_weather_aware.yaml
  ```

Training scripts save the best mIoU checkpoint in `logs/checkpoints/` (or a custom path specified via `save_path` in the config).

## TensorBoard Visualization

- By default, training loss, validation mIoU, learning rate, and other metrics are logged to `logs/tensorboard/<experiment_name>`.
- You can change the output path via `tensorboard_log_dir` in the config.
- Use `tensorboard --logdir logs/tensorboard` (or your custom directory) to view training and evaluation curves in real-time.

## Evaluation and Visualization

- **Overall mIoU**: Use `evaluate` from `eval/eval_segmentation.py`, or reuse it directly in training scripts.
- **Per-weather mIoU**: Call `evaluate_per_weather`; the DataLoader must return `batch["weather"]` field.
- **Prediction visualization**: `utils/visualization.py` provides `save_visual_comparison` to generate "RGB / Model Predictions / GT" comparison images.
- **Quick visualization script**: Use `python scripts/visualize_predictions.py --cfg <config> --ckpt <checkpoint> --list <split_file> --sample-idx <idx> --out <output.png>` to visualize predictions.

## Directory Structure

```
configs/                # YAML configuration files
datasets/               # CARLA dataset definition
models/                 # DeepLabV3 wrapper
samplers/               # Weather-weighted sampler
train/                  # Training scripts (baseline & weather-aware)
eval/                   # mIoU evaluation utilities
utils/                  # Common transforms, metrics, runner, visualization
scripts/                # Data collection and splitting scripts
  - collect_multi_weather.py    # CARLA data collection
  - prepare_splits.py            # Generate train/val/test splits
  - visualize_predictions.py    # Visualization tool
data/                   # Placeholder directory for CARLA data
logs/                   # Logs & checkpoint output
```

## Key Features

- **Weather-aware curriculum learning**: Gradually introduce difficult weather conditions (clear → soft rain → hard rain + night)
- **Weighted sampling**: Balance samples from different weather conditions during training
- **Weather-specific augmentation**: Apply weather-appropriate data augmentation based on sample weather type
- **Per-weather evaluation**: Track mIoU separately for each weather condition
- **GroupNorm support**: Use GroupNorm instead of BatchNorm for better performance with small batch sizes

## Configuration Notes

- Default `image_size` is `[512, 512]` (H, W), can be modified in config.
- Masks should use semantic class indices (0 ~ num_classes-1), with background/ignore value defaulting to 255.
- You can customize staged training by adding a `curriculum` field in YAML (see `train/train_weather_aware.py` for details).
- The curriculum learning approach uses three stages:
  - Stage 1: Clear weather only
  - Stage 2: Clear + soft rain
  - Stage 3: All weathers (clear + soft rain + hard rain + night)

## Requirements

- Python 3.10+
- PyTorch (with CUDA support recommended)
- CARLA Simulator (for data collection)
- See `requirements.txt` or install dependencies as shown in Quick Start
