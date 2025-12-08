from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from utils.transforms import get_common_transforms, get_weather_specific_aug

LOGGER = logging.getLogger(__name__)

# Mapping from CARLA CityScapesPalette pixel values to CityScapes labels
# These are the original pixel values used by CARLA when saving semantic segmentation images
CARLA_PIXEL_TO_LABEL = {
    0: 0,      # road
    1: 1,      # sidewalk
    2: 2,      # building
    3: 3,      # wall
    4: 4,      # fence
    5: 5,      # pole
    6: 6,      # traffic light
    7: 7,      # traffic sign
    8: 8,      # vegetation
    9: 9,      # terrain
    10: 10,    # sky
    11: 11,    # person
    12: 12,    # rider
    13: 13,    # car
    14: 14,    # truck
    15: 15,    # bus
    16: 16,    # train
    17: 17,    # motorcycle
    18: 18,    # bicycle
    # void/background values will be mapped to ignore_index (255)
}


class CarlaWeatherDataset(Dataset):
    """CARLA semantic segmentation dataset with weather labels and optional augmentation."""

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
            raise RuntimeError(f"No samples available for {self.split} split, please check paths or list file")
        self._active_indices = list(range(len(self._all_samples)))
        self._active_weathers = set(weather.lower() for weather in weather_list) if weather_list else None

    # ------------------------------------------------------------------
    def _build_index(self, weather_list: Optional[Sequence[str]]):
        allowed = {w.lower() for w in weather_list} if weather_list else None
        # If macro categories are provided (e.g., "clear", "rain", "night"), try to expand them
        # to specific weather preset names based on actual subdirectories under data_root/carla
        # (e.g., clear -> clearnoon, clearsunset, clearnight)
        carla_dir = self.data_root / "carla"
        if allowed and carla_dir.exists():
            allowed = self._expand_allowed(allowed, carla_dir)
        samples = []
        if self.list_file:
            samples = self._read_list_file(self.list_file, allowed)
        else:
            carla_dir = self.data_root / "carla"
            if not carla_dir.exists():
                raise FileNotFoundError(f"Directory not found: {carla_dir}, please ensure data is placed correctly")
            samples = self._scan_folder(carla_dir, allowed)
        return samples

    def _expand_allowed(self, allowed: set[str], carla_dir: Path) -> set[str]:
        """
        Expand macro categories and convert weather names to lowercase to match splits file format.
        Examples:
        - allowed={'clear','rain'} and 'ClearNoon' exists -> returns {'clear','rain','clearnoon'}
        - allowed={'clearnoon',...} -> returns {'clearnoon',...} (already lowercase)
        """
        expanded = set(allowed)
        try:
            for p in carla_dir.iterdir():
                if not p.is_dir():
                    continue
                dir_name_lower = p.name.lower()
                
                # For each allowed token (whether macro or specific name)
                for token in list(allowed):
                    # 1. If token is a macro category (e.g., "clear"), check if directory name contains it
                    if token in dir_name_lower:
                        expanded.add(dir_name_lower)
                        break
                    # 2. If token is already a directory name (e.g., "clearnoon"), add directly
                    elif token == dir_name_lower:
                        expanded.add(dir_name_lower)
                        break
        except Exception:
            # If traversal fails (e.g., permission/IO), keep original allowed
            pass
        return expanded

    def _read_list_file(self, list_file: Path, allowed: Optional[set[str]]):
        samples: List[Dict[str, str]] = []
        with list_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Format error in list file {list_file} at line: {line}")
                img_path, mask_path, weather = parts[0], parts[1], parts[2]
                weather_lower = weather.lower()
                if allowed and weather_lower not in allowed:
                    continue
                img_path = self._resolve_path(img_path)
                mask_path = self._resolve_path(mask_path)
                if not img_path.exists() or not mask_path.exists():
                    LOGGER.warning("Skipping non-existent sample: %s", line)
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
                LOGGER.warning("%s missing images or masks subdirectory, skipping", weather_dir)
                continue
            mask_lookup = self._build_mask_lookup(mask_dir)
            for img_path in sorted(img_dir.glob("*")):
                if not img_path.is_file():
                    continue
                stem = img_path.stem
                mask_path = mask_lookup.get(stem)
                if mask_path is None:
                    LOGGER.warning("%s missing corresponding mask, ignored", img_path)
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
        
        # Map CARLA CityScapesPalette pixel values to continuous class labels
        mask_array = np.array(mask, dtype=np.uint8)
        remapped_mask = self._remap_mask_values(mask_array)
        mask = Image.fromarray(remapped_mask, mode='L')

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

    @staticmethod
    def _remap_mask_values(mask_array: np.ndarray, ignore_index: int = 255, num_classes: int = 20) -> np.ndarray:
        """
        Map CARLA CityScapesPalette / trainId pixel values to continuous class labels [0, num_classes-1],
        set others to ignore_index.
        Note: CARLA's trainId may contain classes >19 (e.g., train/subway), which should be explicitly ignored.
        """
        remapped = np.full_like(mask_array, ignore_index, dtype=np.uint8)

        # Map known classes
        for pixel_val, label_idx in CARLA_PIXEL_TO_LABEL.items():
            remapped[mask_array == pixel_val] = label_idx

        # Force values beyond num_classes to ignore_index to prevent large class indices during evaluation
        remapped[remapped >= num_classes] = ignore_index
        return remapped

    # ------------------------------------------------------------------
    def set_active_weathers(self, weather_list: Optional[Iterable[str]]):
        """
        Activate a subset based on the provided weather list (may include macro categories like clear/rain/night).
        Macro categories are mapped to specific weather directories via substring matching (e.g., clear -> clearnoon/clearsunset/...).
        """
        if weather_list is None:
            self._active_indices = list(range(len(self._all_samples)))
            self._active_weathers = None
            return

        allowed_tokens = {w.lower() for w in weather_list}

        def _match(weather: str) -> bool:
            # Exact match or substring match (supports macro categories like clear/rain/night)
            for token in allowed_tokens:
                if weather == token or token in weather or weather in token:
                    return True
            return False

        matched_indices = []
        matched_weathers = set()
        for i, sample in enumerate(self._all_samples):
            weather = sample["weather"]
            if _match(weather):
                matched_indices.append(i)
                matched_weathers.add(weather)

        if not matched_indices:
            raise RuntimeError(f"No samples matched for weather_list={allowed_tokens}")

        self._active_indices = matched_indices
        self._active_weathers = matched_weathers

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
