"""
ViT image classifier built on top of HuggingFace AutoModelForImageClassification.

freeze_backbone()  — freeze everything except the classifier head.
unfreeze_blocks()  — selectively unfreeze transformer blocks by index.
unfreeze_all()     — lift all freezes for Phase C top-off.
"""

from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch.nn as nn


MODEL_NAME = "google/vit-base-patch16-384"


def build_vit(num_classes: int = 5, model_name: str = MODEL_NAME) -> nn.Module:
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,   # replaces the 1000-class head
    )
    return model


def get_processor(model_name: str = MODEL_NAME) -> AutoImageProcessor:
    return AutoImageProcessor.from_pretrained(model_name)


def freeze_backbone(model: nn.Module) -> None:
    """Phase A: only the classifier head is trainable."""
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


def unfreeze_blocks(model: nn.Module, start: int = 6) -> None:
    """Phase B: unfreeze transformer blocks [start, 11] + classifier head."""
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        elif "encoder.layer." in name:
            # name like: vit.encoder.layer.8.attention...
            parts = name.split(".")
            try:
                block_idx = int(parts[parts.index("layer") + 1])
                param.requires_grad = block_idx >= start
            except (ValueError, IndexError):
                param.requires_grad = False
        else:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Phase C: unfreeze everything."""
    for param in model.parameters():
        param.requires_grad = True


def trainable_count(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Trainable: {trainable:,} / {total:,}"
