from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset

from utils.transforms import get_common_transforms, get_weather_specific_aug

LOGGER = logging.getLogger(__name__)


class CarlaWeatherDataset(Dataset):
    """CARLA 语义分割数据集，包含天气标签与可选增强。"""

    def __init__(
        self,
        data_root: str | os.PathLike,
        image_size: Sequence[int | float] | int = (512, 512),
        weather_list: Optional[Sequence[str]] = None,
        list_file: Optional[str | os.PathLike] = None,
        split: Optional[str] = None,
        is_train: bool = True,
        use_weather_aug: bool = False,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split or ("train" if is_train else "val")
        self.is_train = is_train
        self.use_weather_aug = use_weather_aug and is_train
        self.common_transform = get_common_transforms(image_size, is_train=is_train)
        self.list_file = Path(list_file).expanduser().resolve() if list_file else None

        self._all_samples = self._build_index(weather_list)
        if not self._all_samples:
            raise RuntimeError(f"{self.split} split 无可用样本，请检查路径或列表文件")
        self._active_indices = list(range(len(self._all_samples)))
        self._active_weathers = set(weather.lower() for weather in weather_list) if weather_list else None

    # ------------------------------------------------------------------
    def _build_index(self, weather_list: Optional[Sequence[str]]):
        allowed = {w.lower() for w in weather_list} if weather_list else None
        samples = []
        if self.list_file:
            samples = self._read_list_file(self.list_file, allowed)
        else:
            carla_dir = self.data_root / "carla"
            if not carla_dir.exists():
                raise FileNotFoundError(f"未找到 {carla_dir}，请确保数据已放置正确")
            samples = self._scan_folder(carla_dir, allowed)
        return samples

    def _read_list_file(self, list_file: Path, allowed: Optional[set[str]]):
        samples: List[Dict[str, str]] = []
        with list_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"列表文件 {list_file} 中存在格式错误行: {line}")
                img_path, mask_path, weather = parts[0], parts[1], parts[2]
                weather_lower = weather.lower()
                if allowed and weather_lower not in allowed:
                    continue
                img_path = self._resolve_path(img_path)
                mask_path = self._resolve_path(mask_path)
                if not img_path.exists() or not mask_path.exists():
                    LOGGER.warning("跳过不存在的样本: %s", line)
                    continue
                samples.append(
                    {
                        "image": img_path,
                        "mask": mask_path,
                        "weather": weather_lower,
                    }
                )
        return samples

    def _scan_folder(self, carla_dir: Path, allowed: Optional[set[str]]):
        samples: List[Dict[str, Path]] = []
        for weather_dir in sorted(p for p in carla_dir.iterdir() if p.is_dir()):
            weather = weather_dir.name.lower()
            if allowed and weather not in allowed:
                continue
            img_dir = weather_dir / "images"
            mask_dir = weather_dir / "masks"
            if not img_dir.exists() or not mask_dir.exists():
                LOGGER.warning("%s 缺少 images 或 masks 子目录，跳过", weather_dir)
                continue
            mask_lookup = self._build_mask_lookup(mask_dir)
            for img_path in sorted(img_dir.glob("*")):
                if not img_path.is_file():
                    continue
                stem = img_path.stem
                mask_path = mask_lookup.get(stem)
                if mask_path is None:
                    LOGGER.warning("%s 缺少对应的mask，已忽略", img_path)
                    continue
                samples.append({"image": img_path, "mask": mask_path, "weather": weather})
        return samples

    @staticmethod
    def _build_mask_lookup(mask_dir: Path):
        lookup: Dict[str, Path] = {}
        for mask_path in mask_dir.glob("*"):
            if mask_path.is_file():
                lookup[mask_path.stem] = mask_path
        return lookup

    def _resolve_path(self, path_str: str):
        path = Path(path_str)
        if not path.is_absolute():
            path = (self.data_root / path_str).resolve()
        return path

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self._active_indices)

    def __getitem__(self, idx: int):  # type: ignore[override]
        real_idx = self._active_indices[idx]
        sample = self._all_samples[real_idx]
        image = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")

        if self.use_weather_aug:
            aug = get_weather_specific_aug(sample["weather"])
            image, mask = aug(image, mask)

        image_tensor, mask_tensor = self.common_transform(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor.long(),
            "weather": sample["weather"],
            "image_path": str(sample["image"]),
            "mask_path": str(sample["mask"]),
        }

    # ------------------------------------------------------------------
    def set_active_weathers(self, weather_list: Optional[Iterable[str]]):
        if weather_list is None:
            self._active_indices = list(range(len(self._all_samples)))
            self._active_weathers = None
            return
        allowed = {w.lower() for w in weather_list}
        self._active_indices = [i for i, s in enumerate(self._all_samples) if s["weather"] in allowed]
        if not self._active_indices:
            raise RuntimeError(f"当前 weather_list={allowed} 未匹配到样本")
        self._active_weathers = allowed

    def get_weather_sequence(self) -> List[str]:
        return [self._all_samples[i]["weather"] for i in self._active_indices]

    def weather_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for weather in self.get_weather_sequence():
            hist[weather] = hist.get(weather, 0) + 1
        return hist

    def enable_weather_aug(self, flag: bool):
        self.use_weather_aug = flag and self.is_train

    def __repr__(self) -> str:
        info = {
            "split": self.split,
            "total_samples": len(self._all_samples),
            "active_samples": len(self._active_indices),
            "use_weather_aug": self.use_weather_aug,
        }
        return f"CarlaWeatherDataset({info})"


__all__ = ["CarlaWeatherDataset"]
