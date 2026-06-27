"""
Phase 6 — Official APTOS test-set evaluation.

Runs inference on test.csv + test_images/ using the best ViT checkpoint
(or an ensemble), verifies label mapping before any computation, then
saves metrics, confusion matrix, and per-class F1 bar chart.

IMPORTANT: label_remap unit tests run automatically before inference.
Any remap failure aborts the script to prevent the negative-QWK bug.

Usage (single model):
    python -m src.eval.evaluate_test \
        --config configs/base.yaml \
        --checkpoint checkpoints/vit/best_phaseC.pt \
        --out_dir experiments/

Usage (ensemble of prob .npy files already saved):
    python -m src.eval.evaluate_test \
        --config configs/base.yaml \
        --probs_npy checkpoints/vit/test_probs.npy checkpoints/resnet/test_probs.npy \
        --labels_npy data/test_labels.npy \
        --out_dir experiments/
"""

import argparse
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score

# --- run remap tests BEFORE anything else ---
from src.eval import label_remap
label_remap.test_remap_is_correct()
label_remap.test_remap_vectorized()
label_remap.test_remap_preserves_qwk_sign()

from src.data.build_dataframe import build_test_df, build_train_df
from src.data.image_audit import audit
from src.data.dataset import RetinopathyDataset
from src.data.augmentations import valid_transforms
from src.models.vit_model import build_vit
from src.eval.metrics import compute_metrics, print_metrics
from src.eval.label_remap import remap, APTOS_CLASSES


CLASS_NAMES = [APTOS_CLASSES[i] for i in range(5)]


# ---------------------------------------------------------------------------
# inference helpers
# ---------------------------------------------------------------------------

def infer_vit(
    checkpoint: str,
    df,
    cfg: dict,
    device: torch.device,
) -> np.ndarray:
    """Return softmax probs (N, 5) from a ViT checkpoint."""
    model = build_vit(num_classes=cfg["data"]["num_classes"]).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_size = cfg["data"]["image_size"]
    ds = RetinopathyDataset(df, image_size=img_size,
                            transform=valid_transforms(img_size), has_labels=False)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, num_workers=cfg["training"]["num_workers"])

    all_probs = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Inference"):
            logits = model(imgs.to(device)).logits
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(all_probs)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def save_confusion_matrix(labels, preds, out_path: str) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(5)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {out_path}")


def save_f1_bar(labels, preds, out_path: str) -> None:
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0, labels=list(range(5)))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASS_NAMES, per_class_f1, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score (Test Set)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved per-class F1 chart → {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(cfg: dict, args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load or infer probs ----
    if args.probs_npy:
        probs_list = [np.load(p) for p in args.probs_npy]
        probs = np.mean(probs_list, axis=0)
        labels = np.load(args.labels_npy)
    else:
        # Need ground-truth labels — use train split for known-label evaluation,
        # or test.csv if competition labels are available.
        if args.use_valid:
            from src.data.split import stratified_split
            df = audit(build_train_df(cfg["data"]["root"]))
            _, eval_df = stratified_split(df,
                                          valid_fraction=cfg["data"]["valid_fraction"],
                                          random_seed=cfg["data"]["random_seed"])
            labels = eval_df["label"].values
        else:
            eval_df = audit(build_test_df(cfg["data"]["root"]))
            labels = None  # test labels not available in Kaggle competition

        probs = infer_vit(args.checkpoint, eval_df, cfg, device)

        # Save probs for ensemble reuse
        np.save(str(out_dir / "test_probs_vit.npy"), probs)

    # ---- apply label remap ----
    raw_preds = probs.argmax(axis=1)
    preds = remap(raw_preds)

    if labels is not None:
        labels = remap(np.asarray(labels))   # remap ground truth too
        m = compute_metrics(labels, preds, probs)
        print("\n=== APTOS Test Evaluation ===")
        print_metrics(m, prefix="test")

        # Save metrics
        metrics_path = out_dir / "test_metrics.txt"
        with open(metrics_path, "w") as f:
            for k, v in m.items():
                f.write(f"{k}: {v}\n")
        print(f"  Saved metrics → {metrics_path}")

        save_confusion_matrix(labels, preds, str(out_dir / "confusion_matrix.png"))
        save_f1_bar(labels, preds, str(out_dir / "per_class_f1.png"))
    else:
        # Kaggle submission mode — no ground truth
        sub_path = out_dir / "submission.csv"
        import pandas as pd
        sub = pd.DataFrame({"id_code": eval_df["image_id"], "diagnosis": preds})
        sub.to_csv(sub_path, index=False)
        print(f"  Saved Kaggle submission → {sub_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--probs_npy", nargs="+", default=None, type=str)
    parser.add_argument("--labels_npy", default=None, type=str)
    parser.add_argument("--out_dir", default="experiments/", type=str)
    parser.add_argument("--use_valid", action="store_true",
                        help="Evaluate on validation split instead of test set")
    args = parser.parse_args()

    if args.checkpoint is None and args.probs_npy is None:
        parser.error("Provide either --checkpoint or --probs_npy")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args)
