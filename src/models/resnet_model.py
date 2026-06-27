"""
ResNet-50 classifier in PyTorch via timm.
Mirrors the vit_model.py freeze/unfreeze API so train_resnet.py
can reuse the same run_epoch / train_phase helpers from train_vit.py.
"""

import timm
import torch.nn as nn


RESNET_NAME = "resnet50"


def build_resnet(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(RESNET_NAME, pretrained=pretrained, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the final classification head (model.fc)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc")


def unfreeze_layers(model: nn.Module, start_layer: int = 3) -> None:
    """
    Unfreeze ResNet layer groups from start_layer onward.
    ResNet-50 has 4 layer groups: layer1, layer2, layer3, layer4.
    start_layer=3 unfreezes layer3, layer4, and fc.
    start_layer=1 unfreezes everything.
    """
    unfreeze_names = {f"layer{i}" for i in range(start_layer, 5)} | {"fc"}
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(n) for n in unfreeze_names)


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def trainable_count(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Trainable: {trainable:,} / {total:,}"
