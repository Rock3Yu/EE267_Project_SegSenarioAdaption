from __future__ import annotations

import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

try:  # torchvision>=0.13
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
except ImportError:  # Compatible with older versions
    DeepLabV3_ResNet50_Weights = None


def convert_bn_to_gn(module: nn.Module, num_groups: int = 32):
    """
    Convert all BatchNorm layers in the model to GroupNorm to avoid issues with small batch sizes.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups, num_channels))
        else:
            convert_bn_to_gn(child, num_groups)
    return module


def get_deeplabv3_resnet50(num_classes: int, pretrained: bool = True, use_gn: bool = False) -> nn.Module:
    """
    Build DeepLabV3-ResNet50 and replace classification head.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        use_gn: Whether to use GroupNorm instead of BatchNorm (for small batch size scenarios)
    """

    weights = None
    if DeepLabV3_ResNet50_Weights is not None:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        model = deeplabv3_resnet50(weights=weights)
    else:
        model = deeplabv3_resnet50(pretrained=pretrained)

    # If using GroupNorm, convert all BatchNorm layers
    if use_gn:
        model = convert_bn_to_gn(model, num_groups=32)

    classifier_in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(classifier_in_channels, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        aux_in_channels = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

    return model


__all__ = ["get_deeplabv3_resnet50"]
