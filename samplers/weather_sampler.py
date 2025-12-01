from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import WeightedRandomSampler

DEFAULT_WEIGHTS: Dict[str, float] = {
    "clear": 1.0,
    "rain": 2.0,
    "night": 3.0,
    "fog": 3.0,
}


def build_weather_balanced_sampler(dataset, weight_map: Optional[Dict[str, float]] = None) -> WeightedRandomSampler:
    """根据天气分布创建WeightedRandomSampler。"""

    if not hasattr(dataset, "get_weather_sequence"):
        raise AttributeError("dataset 必须实现 get_weather_sequence()")

    weights_cfg = {**DEFAULT_WEIGHTS, **(weight_map or {})}
    weather_sequence = dataset.get_weather_sequence()
    if not weather_sequence:
        raise RuntimeError("当前数据集无样本，无法创建采样器")

    weights = torch.tensor([weights_cfg.get(w, 1.0) for w in weather_sequence], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


__all__ = ["build_weather_balanced_sampler", "DEFAULT_WEIGHTS"]
