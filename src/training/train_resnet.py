"""
ResNet-50 training in PyTorch (via timm) — two phases.

Phase 1 (head):     freeze backbone, train fc layer only.
Phase 2 (finetune): unfreeze layer3+layer4+fc, lower LR.
Phase 3 (full):     unfreeze all, very low LR, cosine decay, early stopping.

Reuses run_epoch / train_phase / save_checkpoint from train_vit.py.

Usage:
    python -m src.training.train_resnet --config configs/base.yaml
    python -m src.training.train_resnet --config configs/colab.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.build_dataframe import build_train_df
from src.data.image_audit import audit
from src.data.split import stratified_split
from src.data.dataset import RetinopathyDataset
from src.data.augmentations import train_transforms, valid_transforms
from src.models.resnet_model import (
    build_resnet, freeze_backbone, unfreeze_layers, unfreeze_all, trainable_count,
)
from src.training.class_weights import compute_class_weights
from src.training.train_vit import run_epoch, save_checkpoint, train_phase


def main(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = audit(build_train_df(cfg["data"]["root"]))
    train_df, valid_df = stratified_split(
        df,
        valid_fraction=cfg["data"]["valid_fraction"],
        random_seed=cfg["data"]["random_seed"],
    )
    print(f"Train={len(train_df)}  Valid={len(valid_df)}")

    img_size = cfg["data"]["image_size"]
    bs  = cfg["training"]["batch_size"]
    nw  = cfg["training"]["num_workers"]

    train_loader = DataLoader(
        RetinopathyDataset(train_df, img_size, train_transforms(img_size)),
        batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
    )
    valid_loader = DataLoader(
        RetinopathyDataset(valid_df, img_size, valid_transforms(img_size)),
        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
    )

    model = build_resnet(num_classes=cfg["data"]["num_classes"]).to(device)
    class_weights = compute_class_weights(train_df["label"], device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["vit"]["label_smoothing"],
    )
    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    ckpt_dir = str(Path(cfg["paths"]["checkpoints"]) / "resnet")
    patience = cfg["vit"]["early_stop_patience"]
    cosine   = torch.optim.lr_scheduler.CosineAnnealingLR

    # Phase 1 — head only
    print("\n=== ResNet Phase 1: fc head only ===")
    freeze_backbone(model)
    print(trainable_count(model))
    opt_a = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["resnet"]["head_lr"],
    )
    train_phase("phase1", model, train_loader, valid_loader, criterion,
                opt_a, None, cfg["resnet"]["head_epochs"], device, scaler, ckpt_dir)

    # Phase 2 — unfreeze layer3 + layer4 + fc
    print("\n=== ResNet Phase 2: layer3/4 + fc ===")
    unfreeze_layers(model, start_layer=3)
    print(trainable_count(model))
    opt_b = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["resnet"]["finetune_lr"] * 10,
        weight_decay=1e-4,
    )
    epochs_b = cfg["resnet"]["finetune_epochs"] // 2
    sched_b  = cosine(opt_b, T_max=epochs_b)
    train_phase("phase2", model, train_loader, valid_loader, criterion,
                opt_b, sched_b, epochs_b, device, scaler, ckpt_dir)

    # Phase 3 — full unfreeze, very low LR, early stopping
    print("\n=== ResNet Phase 3: full unfreeze ===")
    unfreeze_all(model)
    print(trainable_count(model))
    opt_c   = torch.optim.AdamW(model.parameters(), lr=cfg["resnet"]["finetune_lr"], weight_decay=1e-4)
    epochs_c = cfg["resnet"]["finetune_epochs"] - epochs_b
    sched_c  = cosine(opt_c, T_max=epochs_c)
    best_qwk = train_phase(
        "phase3", model, train_loader, valid_loader, criterion,
        opt_c, sched_c, epochs_c, device, scaler, ckpt_dir,
        early_stop_patience=patience,
    )

    print(f"\nResNet training complete. Best QWK = {best_qwk:.4f}")
    print(f"Best checkpoint: {ckpt_dir}/best_phase3.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", type=str)
    parser.add_argument("--data_root", default=None, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data"]["root"] = args.data_root

    main(cfg)
