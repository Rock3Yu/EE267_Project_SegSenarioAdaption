from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def _fast_hist(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    """
    Fast confusion matrix computation with defensive checks:
    - Clamp out-of-bound labels to ignore_index to prevent huge memory allocation in bincount.
    - Raise error for abnormally large num_classes.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if num_classes > 512:
        raise ValueError(f"num_classes={num_classes} is too large, likely a configuration error. Aborting to prevent memory explosion.")

    # Filter out-of-bound labels/predictions
    target = target.astype(np.int64)
    pred = pred.astype(np.int64)
    target[(target < 0) | (target >= num_classes)] = ignore_index
    pred[(pred < 0) | (pred >= num_classes)] = ignore_index

    mask = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    if ignore_index >= 0:
        mask &= (target != ignore_index) & (pred != ignore_index)

    if not np.any(mask):
        return np.zeros((num_classes, num_classes), dtype=np.float64)

    hist = np.bincount(
        num_classes * target[mask] + pred[mask],
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist


def per_class_iou(conf_mat: np.ndarray) -> np.ndarray:
    diag = np.diag(conf_mat)
    denom = conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - diag
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = diag / np.maximum(denom, 1e-6)
    # Set classes with zero denominator to NaN so they are ignored in mIoU calculation
    iou[denom == 0] = np.nan
    return iou


class SegmentationMetricMeter:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_np = preds.detach().cpu().numpy().astype(np.int64)
        targets_np = targets.detach().cpu().numpy().astype(np.int64)
        self.conf_mat += _fast_hist(preds_np.flatten(), targets_np.flatten(), self.num_classes, self.ignore_index)

    def get_scores(self) -> Dict[str, np.ndarray | float]:
        class_iou = per_class_iou(self.conf_mat)
        valid = ~np.isnan(class_iou)
        miou = float(np.nanmean(class_iou[valid])) if valid.any() else 0.0
        
        # Add diagnostic information
        return {
            "per_class_iou": class_iou,
            "mIoU": miou,
            "confusion_matrix": self.conf_mat,
            "num_valid_classes": int(valid.sum()),  # Number of classes that actually appear
        }


def compute_batch_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> Tuple[np.ndarray, float]:
    meter = SegmentationMetricMeter(num_classes=num_classes, ignore_index=ignore_index)
    meter.update(preds, targets)
    scores = meter.get_scores()
    return scores["per_class_iou"], float(scores["mIoU"])


__all__ = ["SegmentationMetricMeter", "compute_batch_iou", "per_class_iou"]
