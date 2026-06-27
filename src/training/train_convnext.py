"""
Phase 5d — ConvNeXt-Base training script (mirrors train_vit.py structure).

Phases:
  A : head only
  B : unfreeze stages 2-3
  C : full unfreeze, cosine decay, early stopping on QWK

Usage:
    python -m src.training.train_convnext --config configs/base.yaml
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
from src.models.convnext_model import (
    build_convnext, freeze_backbone, unfreeze_stages, unfreeze_all, trainable_count,
)
from src.training.class_weights import compute_class_weights
from src.training.train_vit import run_epoch, save_checkpoint, train_phase
from src.eval.metrics import print_metrics


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
    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]

    train_loader = DataLoader(
        RetinopathyDataset(train_df, img_size, train_transforms(img_size)),
        batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
    )
    valid_loader = DataLoader(
        RetinopathyDataset(valid_df, img_size, valid_transforms(img_size)),
        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
    )

    model = build_convnext(num_classes=cfg["data"]["num_classes"]).to(device)
    class_weights = compute_class_weights(train_df["label"], device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["vit"]["label_smoothing"],  # reuse same smoothing
    )
    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    ckpt_dir = str(Path(cfg["paths"]["checkpoints"]) / "convnext")
    patience = cfg["vit"]["early_stop_patience"]

    # Phase A
    print("\n=== ConvNeXt Phase A: head only ===")
    freeze_backbone(model)
    print(trainable_count(model))
    opt_a = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["vit"]["phase_a_lr"],
    )
    train_phase("phaseA", model, train_loader, valid_loader, criterion,
                opt_a, None, cfg["vit"]["phase_a_epochs"], device, scaler, ckpt_dir)

    # Phase B
    print("\n=== ConvNeXt Phase B: stages 2-3 ===")
    unfreeze_stages(model, start_stage=2)
    print(trainable_count(model))
    opt_b = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["vit"]["phase_b_lr"], weight_decay=1e-4,
    )
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=cfg["vit"]["phase_b_epochs"])
    train_phase("phaseB", model, train_loader, valid_loader, criterion,
                opt_b, sched_b, cfg["vit"]["phase_b_epochs"], device, scaler, ckpt_dir)

    # Phase C
    print("\n=== ConvNeXt Phase C: full unfreeze ===")
    unfreeze_all(model)
    print(trainable_count(model))
    opt_c = torch.optim.AdamW(model.parameters(), lr=cfg["vit"]["phase_c_lr"], weight_decay=1e-4)
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=cfg["vit"]["phase_c_epochs"])
    best_qwk = train_phase(
        "phaseC", model, train_loader, valid_loader, criterion,
        opt_c, sched_c, cfg["vit"]["phase_c_epochs"], device, scaler, ckpt_dir,
        early_stop_patience=patience,
    )

    print(f"\nConvNeXt training complete. Best QWK = {best_qwk:.4f}")
    print(f"Best checkpoint: {ckpt_dir}/best_phaseC.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
