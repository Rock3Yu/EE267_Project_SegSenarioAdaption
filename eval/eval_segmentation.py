from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

from utils.metrics import SegmentationMetricMeter


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """整体 mIoU 评估。"""

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
    return float(scores["mIoU"])


def evaluate_per_weather(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """按天气分别计算 mIoU。"""

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
