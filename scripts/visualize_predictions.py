#!/usr/bin/env python3
"""
Generate prediction visualization comparison (RGB / Prediction / GT) from checkpoint.

Usage example:
  python scripts/visualize_predictions.py \
    --cfg configs/ours_weather_aware.yaml \
    --ckpt logs/checkpoints/weather_aware_full_v2.pth \
    --list data/splits/val.txt \
    --sample-idx 0 \
    --out vis_compare.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets import CarlaWeatherDataset
from models import get_deeplabv3_resnet50
from utils.runner import load_config  # type: ignore
from utils.visualization import save_visual_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prediction visualization comparison")
    parser.add_argument("--cfg", required=True, help="YAML config path for parsing data paths and num_classes")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint path (contains model_state)")
    parser.add_argument("--list", help="Splits list file (img mask weather); if not provided, uses cfg's val_list")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample to visualize")
    parser.add_argument("--out", default="vis_compare.png", help="Output image path")
    return parser.parse_args()


def _resolve_path(path_str: str | Path, base_dir: Path, project_root: Path) -> Path:
    """Path resolution: prefer project root, fallback to cfg directory."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    # Try relative to project root first
    root_path = (project_root / path).resolve()
    if root_path.exists():
        return root_path
    # Otherwise relative to cfg directory (for backward compatibility)
    return (base_dir / path).resolve()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config to get num_classes / data_root / val_list / use_group_norm
    cfg = load_config(args.cfg)
    num_classes = int(cfg.get("num_classes", 20))
    use_gn = bool(cfg.get("use_group_norm", True))
    base_dir = Path(cfg["cfg_dir"])
    # Project root: parent of cfg_dir (one level above configs/)
    project_root = base_dir.parent
    data_root = _resolve_path(cfg.get("data_root", "./data"), base_dir, project_root)
    list_file = args.list or cfg.get("val_list")
    if not list_file:
        raise RuntimeError("--list not provided and val_list missing in config")
    list_file = _resolve_path(list_file, base_dir, project_root)

    # Dataset (only used to get one sample)
    ds = CarlaWeatherDataset(
        data_root=str(data_root),
        list_file=str(list_file),
        split="val",
        is_train=False,
        use_weather_aug=False,
        image_size=cfg.get("image_size", (512, 512)),
    )
    idx = max(0, min(args.sample_idx, len(ds) - 1))
    sample = ds[idx]
    rgb = sample["image"]  # Already a tensor, normalized
    gt = sample["mask"].numpy()

    # Model
    model = get_deeplabv3_resnet50(num_classes=num_classes, pretrained=False, use_gn=use_gn).to(device)
    ckpt_path = _resolve_path(args.ckpt, base_dir, project_root)
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state.get("model_state", state), strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    with torch.no_grad():
        logits = model(sample["image"].unsqueeze(0).to(device))["out"]
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

    save_visual_comparison(
        rgb=rgb,
        predictions=[("pred", pred)],
        gt_mask=gt,
        output_path=Path(args.out),
        num_classes=num_classes,
    )
    print(f"Saved to {args.out} (sample {idx})")


if __name__ == "__main__":
    main()

