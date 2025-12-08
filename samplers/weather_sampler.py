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
    """Create WeightedRandomSampler based on weather distribution."""

    if not hasattr(dataset, "get_weather_sequence"):
        raise AttributeError("dataset must implement get_weather_sequence()")

    weights_cfg = {**DEFAULT_WEIGHTS, **(weight_map or {})}
    weather_sequence = dataset.get_weather_sequence()
    if not weather_sequence:
        raise RuntimeError("Current dataset has no samples, cannot create sampler")
    # For each sample's weather (e.g., 'clearnoon'), prefer exact key; if not found, try matching macro tokens in weights_cfg
    resolved_weights = []
    for w in weather_sequence:
        if w in weights_cfg:
            resolved_weights.append(float(weights_cfg[w]))
            continue
        # Try token matching: if a key in weights_cfg is a substring of w or w contains that key, match it
        matched = False
        for k, v in weights_cfg.items():
            if k and (k in w or w in k):
                resolved_weights.append(float(v))
                matched = True
                break
        if not matched:
            resolved_weights.append(1.0)

    weights = torch.tensor(resolved_weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


__all__ = ["build_weather_balanced_sampler", "DEFAULT_WEIGHTS"]
