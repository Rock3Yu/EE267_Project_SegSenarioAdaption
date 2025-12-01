from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np
import torch
import yaml
try:  # tensorboard 可能未安装
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


class AverageMeter:
    """简单的平均器，便于统计loss等指标。"""

    def __init__(self, name: str = "avg") -> None:
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _deep_update(base: MutableMapping[str, Any], overrides: Mapping[str, Any]):
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(cfg_path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(cfg_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_cfg = cfg.pop("base_cfg", None)
    if base_cfg:
        base_path = Path(base_cfg)
        if not base_path.is_absolute():
            base_path = (cfg_path.parent / base_path).resolve()
        base_dict = load_config(base_path)
        cfg = _deep_update(base_dict, cfg)

    cfg["cfg_path"] = str(cfg_path)
    cfg["cfg_dir"] = str(cfg_path.parent)
    return cfg


def create_optimizer(model: torch.nn.Module, cfg: Mapping[str, Any]):
    opt_name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("lr", 2e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    if opt_name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def create_summary_writer(
    log_dir: str | Path,
    experiment_name: Optional[str] = None,
    sub_dir: Optional[str] = None,
) -> SummaryWriter:
    """
    创建 TensorBoard SummaryWriter，自动处理目录。
    """

    if SummaryWriter is None:
        raise RuntimeError("当前环境未安装 tensorboard，执行 `pip install tensorboard` 即可启用记录功能")
    base = Path(log_dir).expanduser().resolve()
    parts = [p for p in [experiment_name, sub_dir] if p]
    writer_dir = base.joinpath(*parts) if parts else base
    writer_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(writer_dir))


__all__ = [
    "AverageMeter",
    "seed_everything",
    "save_checkpoint",
    "load_config",
    "create_optimizer",
    "create_summary_writer",
]
