"""
Phase 5c — 5-fold stratified cross-validation ensemble for the ViT.

Each fold trains a full Phase A→B→C model. Fold OOF (out-of-fold) logits
are saved to disk and averaged at the end to compute ensemble QWK.

Usage:
    python -m src.training.kfold_train \
        --config configs/base.yaml \
        --n_folds 5
"""

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.data.build_dataframe import build_train_df
from src.data.image_audit import audit
from src.data.dataset import RetinopathyDataset
from src.data.augmentations import train_transforms, valid_transforms
from src.models.vit_model import (
    build_vit, freeze_backbone, unfreeze_blocks, unfreeze_all, trainable_count,
)
from src.training.class_weights import compute_class_weights
from src.training.train_vit import run_epoch, save_checkpoint
from src.eval.metrics import compute_metrics, print_metrics


def train_single_fold(
    fold: int,
    train_df,
    valid_df,
    cfg: dict,
    device: torch.device,
    ckpt_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (oof_probs, oof_labels) for this fold."""
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

    model = build_vit(cfg["data"]["num_classes"]).to(device)
    class_weights = compute_class_weights(train_df["label"], device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["vit"]["label_smoothing"],
    )
    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    patience = cfg["vit"]["early_stop_patience"]

    def _phase(name, freeze_fn, lr, epochs, sched_class=None):
        freeze_fn(model)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        sched = sched_class(opt, T_max=epochs) if sched_class else None
        best_qwk, patience_ctr = -1.0, 0
        best_path = str(ckpt_dir / f"fold{fold}_best_{name}.pt")

        for ep in range(1, epochs + 1):
            _, tm = run_epoch(model, train_loader, criterion, opt, device, scaler, train=True)
            _, vm = run_epoch(model, valid_loader, criterion, opt, device, scaler, train=False)
            if sched:
                sched.step()
            print(f"[fold {fold} | {name}] ep {ep}  ", end="")
            print_metrics(vm, prefix="valid")

            if vm["qwk"] > best_qwk:
                best_qwk = vm["qwk"]
                save_checkpoint(model, best_path, vm)
                patience_ctr = 0
            else:
                patience_ctr += 1
                if name == "phaseC" and patience_ctr >= patience:
                    print(f"  Early stop fold {fold} phase C")
                    break

        # Reload best weights for next phase
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR

    _phase("phaseA", freeze_backbone,                    cfg["vit"]["phase_a_lr"], cfg["vit"]["phase_a_epochs"])
    _phase("phaseB", lambda m: unfreeze_blocks(m, 6),   cfg["vit"]["phase_b_lr"], cfg["vit"]["phase_b_epochs"], cosine)
    _phase("phaseC", unfreeze_all,                       cfg["vit"]["phase_c_lr"], cfg["vit"]["phase_c_epochs"], cosine)

    # Collect OOF probabilities
    model.eval()
    oof_probs, oof_labels = [], []
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs = imgs.to(device)
            logits = model(imgs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            oof_probs.append(probs)
            oof_labels.extend(labels.tolist())

    return np.vstack(oof_probs), np.array(oof_labels)


def main(cfg: dict, n_folds: int = 5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = audit(build_train_df(cfg["data"]["root"]))
    labels = df["label"].values

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / "vit_kfold"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg["data"]["random_seed"])

    oof_probs_all  = np.zeros((len(df), cfg["data"]["num_classes"]))
    oof_labels_all = np.zeros(len(df), dtype=int)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, labels), start=1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{n_folds}  train={len(train_idx)}  valid={len(valid_idx)}")
        print(f"{'='*60}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        oof_probs, oof_labels = train_single_fold(
            fold, train_df, valid_df, cfg, device, ckpt_dir
        )

        oof_probs_all[valid_idx]  = oof_probs
        oof_labels_all[valid_idx] = oof_labels

        fold_preds = oof_probs.argmax(axis=1)
        fold_m = compute_metrics(oof_labels, fold_preds, oof_probs)
        print_metrics(fold_m, prefix=f"Fold {fold} OOF")

        # Save fold OOF for later ensemble blending
        np.save(str(ckpt_dir / f"fold{fold}_oof_probs.npy"),  oof_probs)
        np.save(str(ckpt_dir / f"fold{fold}_oof_labels.npy"), oof_labels)

    # Full ensemble OOF
    ensemble_preds = oof_probs_all.argmax(axis=1)
    m = compute_metrics(oof_labels_all, ensemble_preds, oof_probs_all)
    print("\n=== 5-Fold Ensemble OOF ===")
    print_metrics(m, prefix="ensemble")
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", type=str)
    parser.add_argument("--n_folds", default=5, type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.n_folds)
