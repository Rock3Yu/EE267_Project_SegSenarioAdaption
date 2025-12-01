from __future__ import annotations

import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

try:  # torchvision>=0.13
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
except ImportError:  # 兼容旧版本
    DeepLabV3_ResNet50_Weights = None


def get_deeplabv3_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """构建DeepLabV3-ResNet50并替换分类头。"""

    weights = None
    if DeepLabV3_ResNet50_Weights is not None:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        model = deeplabv3_resnet50(weights=weights)
    else:
        model = deeplabv3_resnet50(pretrained=pretrained)

    classifier_in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(classifier_in_channels, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        aux_in_channels = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

    return model


__all__ = ["get_deeplabv3_resnet50"]
