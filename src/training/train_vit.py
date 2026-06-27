"""
ViT 3-phase training script.

Phase A : freeze backbone, train classifier head only.
Phase B : unfreeze blocks 6-11, fine-tune with lower LR.
Phase C : full unfreeze, very low LR, cosine decay, early stopping on QWK.

Usage:
    python -m src.training.train_vit \
        --data_root data/ \
        --checkpoint_dir checkpoints/vit/ \
        --config configs/base.yaml
"""

import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.build_dataframe import build_train_df
from src.data.image_audit import audit
from src.data.split import stratified_split
from src.data.dataset import RetinopathyDataset
from src.data.augmentations import train_transforms, valid_transforms
from src.models.vit_model import (
    build_vit,
    freeze_backbone,
    unfreeze_blocks,
    unfreeze_all,
    trainable_count,
)
from src.training.class_weights import compute_class_weights
from src.eval.metrics import compute_metrics, print_metrics


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(df, image_size, transform, batch_size, num_workers, shuffle):
    ds = RetinopathyDataset(df, image_size=image_size, transform=transform)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
    )


def save_checkpoint(model, path: str, metrics: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, path)
    print(f"  Saved checkpoint → {path}  (QWK={metrics['qwk']:.4f})")


# ---------------------------------------------------------------------------
# one epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train(train)
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False, desc="train" if train else "valid"):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                outputs = model(imgs).logits
                loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# phase runner
# ---------------------------------------------------------------------------

def train_phase(
    phase_name: str,
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    epochs: int,
    device,
    scaler,
    checkpoint_dir: str,
    early_stop_patience: int = 0,
) -> float:
    best_qwk = -1.0
    patience_counter = 0
    best_ckpt = str(Path(checkpoint_dir) / f"best_{phase_name}.pt")

    for epoch in range(1, epochs + 1):
        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        valid_loss, valid_m = run_epoch(model, valid_loader, criterion, optimizer, device, scaler, train=False)

        if scheduler:
            scheduler.step()

        print(f"[{phase_name}] Epoch {epoch}/{epochs}  "
              f"loss={train_loss:.4f}/{valid_loss:.4f}")
        print_metrics(train_m, prefix="train")
        print_metrics(valid_m, prefix="valid")

        # Per-epoch checkpoint
        epoch_ckpt = str(Path(checkpoint_dir) / f"{phase_name}_epoch{epoch:02d}.pt")
        save_checkpoint(model, epoch_ckpt, valid_m)

        if valid_m["qwk"] > best_qwk:
            best_qwk = valid_m["qwk"]
            save_checkpoint(model, best_ckpt, valid_m)
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"  Early stop at epoch {epoch} (no QWK improvement for {early_stop_patience} epochs)")
                break

    print(f"  [{phase_name}] Best QWK = {best_qwk:.4f}")
    return best_qwk


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(cfg: dict) -> None:
    device = get_device()
    print(f"Device: {device}")

    # --- data ---
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

    train_loader = make_loader(train_df, img_size, train_transforms(img_size), bs, nw, shuffle=True)
    valid_loader = make_loader(valid_df, img_size, valid_transforms(img_size), bs, nw, shuffle=False)

    # --- model ---
    model = build_vit(num_classes=cfg["data"]["num_classes"]).to(device)

    # --- loss ---
    class_weights = compute_class_weights(train_df["label"], device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["vit"]["label_smoothing"],
    )

    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    ckpt_dir = cfg["paths"]["checkpoints"] + "vit/"
    patience = cfg["vit"]["early_stop_patience"]

    # -----------------------------------------------------------------------
    # Phase A — head only
    # -----------------------------------------------------------------------
    print("\n=== Phase A: classifier head only ===")
    freeze_backbone(model)
    print(trainable_count(model))

    optimizer_a = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["vit"]["phase_a_lr"],
    )
    train_phase(
        "phaseA", model, train_loader, valid_loader, criterion,
        optimizer_a, scheduler=None,
        epochs=cfg["vit"]["phase_a_epochs"],
        device=device, scaler=scaler,
        checkpoint_dir=ckpt_dir,
    )

    # -----------------------------------------------------------------------
    # Phase B — unfreeze blocks 6-11
    # -----------------------------------------------------------------------
    print("\n=== Phase B: unfreeze blocks 6-11 ===")
    unfreeze_blocks(model, start=6)
    print(trainable_count(model))

    optimizer_b = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["vit"]["phase_b_lr"],
        weight_decay=1e-4,
    )
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, T_max=cfg["vit"]["phase_b_epochs"]
    )
    train_phase(
        "phaseB", model, train_loader, valid_loader, criterion,
        optimizer_b, scheduler_b,
        epochs=cfg["vit"]["phase_b_epochs"],
        device=device, scaler=scaler,
        checkpoint_dir=ckpt_dir,
    )

    # -----------------------------------------------------------------------
    # Phase C — full unfreeze, very low LR, early stopping
    # -----------------------------------------------------------------------
    print("\n=== Phase C: full unfreeze, cosine decay, early stopping ===")
    unfreeze_all(model)
    print(trainable_count(model))

    optimizer_c = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["vit"]["phase_c_lr"],
        weight_decay=1e-4,
    )
    scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_c, T_max=cfg["vit"]["phase_c_epochs"]
    )
    best_qwk = train_phase(
        "phaseC", model, train_loader, valid_loader, criterion,
        optimizer_c, scheduler_c,
        epochs=cfg["vit"]["phase_c_epochs"],
        device=device, scaler=scaler,
        checkpoint_dir=ckpt_dir,
        early_stop_patience=patience,
    )

    print(f"\nViT training complete. Best QWK = {best_qwk:.4f}")
    print(f"Best checkpoint: {ckpt_dir}best_phaseC.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", type=str)
    parser.add_argument("--data_root", default=None, type=str,
                        help="Override data.root from config")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="Override paths.checkpoints from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.checkpoint_dir:
        cfg["paths"]["checkpoints"] = args.checkpoint_dir

    main(cfg)
