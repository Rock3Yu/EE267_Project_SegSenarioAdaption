from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.metrics import SegmentationMetricMeter


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
    verbose: bool = False,
) -> float:
    """
    Overall mIoU evaluation.
    
    Args:
        model: Segmentation model
        dataloader: Validation data loader
        device: Device
        num_classes: Number of classes
        ignore_index: Ignore index
        verbose: Whether to print detailed information
    
    Returns:
        mIoU score
    """
    meter = SegmentationMetricMeter(num_classes=num_classes, ignore_index=ignore_index)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)["out"]
            preds = torch.argmax(logits, dim=1)
            meter.update(preds, masks)
    scores = meter.get_scores()
    
    if verbose:
        print(f"  [Evaluation Details]")
        print(f"    Number of classes that actually appear: {scores.get('num_valid_classes', 'N/A')}/{num_classes}")
        class_iou = scores["per_class_iou"]
        for c in range(min(5, num_classes)):
            iou_val = class_iou[c]
            iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "not present"
            print(f"    Class {c:2d} IoU: {iou_str}")
    
    return float(scores["mIoU"])


def evaluate_per_weather(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """Compute mIoU separately for each weather."""

    meters: Dict[str, SegmentationMetricMeter] = defaultdict(
        lambda: SegmentationMetricMeter(num_classes=num_classes, ignore_index=ignore_index)
    )
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            weathers: Iterable[str] = batch["weather"]
            logits = model(images)["out"]
            preds = torch.argmax(logits, dim=1)
            for idx, weather in enumerate(weathers):
                meters[weather].update(preds[idx : idx + 1], masks[idx : idx + 1])
    weather_scores = {weather: float(meter.get_scores()["mIoU"]) for weather, meter in meters.items()}
    return weather_scores


__all__ = ["evaluate", "evaluate_per_weather"]
