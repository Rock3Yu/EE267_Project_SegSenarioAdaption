#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CARLA weather train/val/test list files")
    parser.add_argument("--data-root", default="./data", help="Data root directory containing carla subdirectory")
    parser.add_argument("--output", default="./data/splits", help="Output directory for train/val/test lists")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relative", action="store_true", help="Use relative paths in lists")
    parser.add_argument(
        "--weathers",
        nargs="*",
        default=None,
        help="Optional: only process specified weathers, e.g., clear rain night",
    )
    return parser.parse_args()


def collect_samples(data_root: Path, weather_filter: List[str] | None):
    carla_dir = data_root / "carla"
    if not carla_dir.exists():
        raise FileNotFoundError(f"Directory not found: {carla_dir}")
    weather_dirs = sorted(p for p in carla_dir.iterdir() if p.is_dir())
    samples: List[Tuple[Path, Path, str]] = []
    for weather_dir in weather_dirs:
        weather = weather_dir.name.lower()
        if weather_filter and weather not in weather_filter:
            continue
        img_dir = weather_dir / "images"
        mask_dir = weather_dir / "masks"
        if not img_dir.exists() or not mask_dir.exists():
            continue
        mask_lookup = {p.stem: p for p in mask_dir.glob("*") if p.is_file()}
        for img_path in img_dir.glob("*"):
            if not img_path.is_file():
                continue
            mask_path = mask_lookup.get(img_path.stem)
            if mask_path is None:
                continue
            samples.append((img_path.resolve(), mask_path.resolve(), weather))
    if not samples:
        raise RuntimeError("No samples collected, please check data paths")
    return samples


def split_samples(samples: List[Tuple[Path, Path, str]], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    rng.shuffle(samples)
    total = len(samples)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    train = samples[:train_end]
    val = samples[train_end:val_end]
    test = samples[val_end:]
    return {"train": train, "val": val, "test": test}


def write_split(split: List[Tuple[Path, Path, str]], out_path: Path, base_dir: Path, use_relative: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for img_path, mask_path, weather in split:
            if use_relative:
                try:
                    img_str = img_path.relative_to(base_dir)
                    mask_str = mask_path.relative_to(base_dir)
                except ValueError:
                    img_str = img_path
                    mask_str = mask_path
            else:
                img_str = img_path
                mask_str = mask_path
            f.write(f"{img_str} {mask_str} {weather}\n")


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output)
    weather_filter = [w.lower() for w in args.weathers] if args.weathers else None
    samples = collect_samples(data_root, weather_filter)
    splits = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    for split_name, split_samples_list in splits.items():
        write_split(
            split_samples_list,
            output_dir / f"{split_name}.txt",
            base_dir=data_root,
            use_relative=args.relative,
        )
    print(f"Generated train/val/test lists in {output_dir}, total {len(samples)} samples")


if __name__ == "__main__":
    main()
