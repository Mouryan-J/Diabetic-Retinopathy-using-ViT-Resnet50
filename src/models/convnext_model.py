"""
Phase 5d — ConvNeXt-Base backbone for ensembling with ViT.

Uses timm's convnext_base pretrained on ImageNet-22k (better than 1k for
fine-grained medical classification).  Exposes the same freeze/unfreeze API
as vit_model.py so it can be dropped into train_vit.py with minor changes.
"""

import timm
import torch.nn as nn


# timm >=0.9 uses dot notation; older versions use underscore names.
# We try the new name first and fall back to the legacy name.
CONVNEXT_NAME_NEW    = "convnext_base.fb_in22k_ft_in1k_384"
CONVNEXT_NAME_LEGACY = "convnext_base_384_in22ft1k"


def build_convnext(num_classes: int = 5, model_name: str | None = None) -> nn.Module:
    if model_name is None:
        try:
            model = timm.create_model(CONVNEXT_NAME_NEW, pretrained=True, num_classes=num_classes)
            print(f"Loaded ConvNeXt: {CONVNEXT_NAME_NEW}")
            return model
        except Exception:
            model = timm.create_model(CONVNEXT_NAME_LEGACY, pretrained=True, num_classes=num_classes)
            print(f"Loaded ConvNeXt: {CONVNEXT_NAME_LEGACY}")
            return model
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the classifier head (model.head)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head")


def unfreeze_stages(model: nn.Module, start_stage: int = 2) -> None:
    """
    Unfreeze ConvNeXt stages from start_stage onward (0-indexed, 4 stages total).
    stage 0 = earliest/most generic; stage 3 = deepest.
    """
    for name, param in model.named_parameters():
        if name.startswith("head"):
            param.requires_grad = True
        elif name.startswith("stages."):
            stage_idx = int(name.split(".")[1])
            param.requires_grad = stage_idx >= start_stage
        else:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def trainable_count(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Trainable: {trainable:,} / {total:,}"
