from __future__ import annotations

import random
from typing import Callable, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _ensure_hw(size: Sequence[int | float] | int) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("image_size only supports integer or sequence like (H, W)")
    height, width = int(size[0]), int(size[1])
    return height, width


def get_common_transforms(
    image_size: Sequence[int | float] | int,
    is_train: bool = True,
) -> Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]]:
    """Return joint transform for image+mask, keeping nearest neighbor interpolation for mask."""

    height, width = _ensure_hw(image_size)
    resize_size = (height, width)
    color_jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)

    def _transform(image: Image.Image, mask: Image.Image):
        image_resized = F.resize(
            image,
            resize_size,
            interpolation=InterpolationMode.BILINEAR,
        )
        mask_resized = F.resize(
            mask,
            resize_size,
            interpolation=InterpolationMode.NEAREST,
        )

        if is_train:
            if random.random() < 0.5:
                image_flipped = F.hflip(image_resized)
                mask_flipped = F.hflip(mask_resized)
            else:
                image_flipped, mask_flipped = image_resized, mask_resized
            image_aug = color_jitter(image_flipped)
        else:
            image_aug, mask_flipped = image_resized, mask_resized

        image_tensor = F.to_tensor(image_aug)
        image_tensor = F.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_array = np.array(mask_flipped, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_array)
        return image_tensor, mask_tensor

    return _transform


def _identity_aug(image: Image.Image, mask: Image.Image):
    return image, mask


def _rain_aug(image: Image.Image, mask: Image.Image):
    img = ImageEnhance.Color(image).enhance(0.9)
    img = ImageEnhance.Brightness(img).enhance(0.95)
    img = ImageEnhance.Contrast(img).enhance(0.95)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img, mask


def _night_aug(image: Image.Image, mask: Image.Image):
    img = ImageEnhance.Brightness(image).enhance(0.55)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Color(img).enhance(0.85)
    return img, mask


def _fog_aug(image: Image.Image, mask: Image.Image):
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1.6))
    haze = Image.new("RGB", image.size, (255, 255, 255))
    img = Image.blend(blurred, haze, alpha=0.18)
    return img, mask


def get_weather_specific_aug(weather: str) -> Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]:
    """Return simple additional augmentation based on weather type."""

    mapping = {
        "clear": _identity_aug,
        "rain": _rain_aug,
        "night": _night_aug,
        "fog": _fog_aug,
    }
    return mapping.get(weather.lower(), _identity_aug)


__all__ = [
    "get_common_transforms",
    "get_weather_specific_aug",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
