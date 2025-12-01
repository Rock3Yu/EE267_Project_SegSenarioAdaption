from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .transforms import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_PALETTE: Sequence[Tuple[int, int, int]] = (
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (0, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (255, 255, 255),
)


def _ensure_palette(num_classes: int, palette: Sequence[Tuple[int, int, int]] | None):
    if palette is None:
        palette = DEFAULT_PALETTE
    if len(palette) >= num_classes:
        return palette
    # 若颜色不足，扩展随机颜色但保持可复现
    rng = np.random.default_rng(seed=1234)
    palette = list(palette)
    while len(palette) < num_classes:
        palette.append(tuple(int(x) for x in rng.integers(0, 255, size=3)))
    return palette


def decode_segmentation(mask: np.ndarray | torch.Tensor, num_classes: int, palette: Sequence[Tuple[int, int, int]] | None = None):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = mask.astype(np.int32)
    palette = _ensure_palette(num_classes, palette)
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in range(num_classes):
        color[mask == idx] = palette[idx]
    return color


def tensor_to_image(tensor: torch.Tensor | np.ndarray, denormalize: bool = True) -> np.ndarray:
    if torch.is_tensor(tensor):
        array = tensor.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            array = np.transpose(array, (1, 2, 0))
    else:
        array = tensor
    if denormalize and array.ndim == 3 and array.shape[2] == 3:
        array = array * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    array = np.clip(array, 0, 1)
    array = (array * 255.0).astype(np.uint8)
    return array


def save_visual_comparison(
    rgb: torch.Tensor | np.ndarray,
    predictions: Iterable[Tuple[str, torch.Tensor | np.ndarray]],
    gt_mask: torch.Tensor | np.ndarray,
    output_path: str | Path,
    num_classes: int,
    palette: Sequence[Tuple[int, int, int]] | None = None,
):
    palette = _ensure_palette(num_classes, palette)
    rgb_img = tensor_to_image(rgb)
    decoded_items: List[Tuple[str, np.ndarray]] = [("RGB", rgb_img)]

    for name, mask in predictions:
        color = decode_segmentation(mask, num_classes=num_classes, palette=palette)
        decoded_items.append((name, color))

    decoded_items.append(("GT", decode_segmentation(gt_mask, num_classes=num_classes, palette=palette)))

    n_cols = len(decoded_items)
    plt.figure(figsize=(4 * n_cols, 4))
    for idx, (title, img) in enumerate(decoded_items, start=1):
        plt.subplot(1, n_cols, idx)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


__all__ = ["save_visual_comparison", "decode_segmentation", "tensor_to_image"]
