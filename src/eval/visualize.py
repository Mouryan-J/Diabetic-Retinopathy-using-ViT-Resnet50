"""
Phase 7 — Visualization suite.

Produces four figures, each saved as a high-res PNG:
  1. per_class_grid.png     — sample fundus images for each severity grade
  2. confusion_matrix.png   — row-normalized heatmap
  3. per_class_f1.png       — bar chart of per-class F1 scores
  4. roc_curves.png         — one-vs-rest ROC curve per class + macro AUC

Usage:
    python -m src.eval.visualize \
        --probs_npy experiments/test_probs_vit.npy \
        --labels_npy data/valid_labels.npy \
        --data_root data/ \
        --out_dir experiments/figures/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src.eval.label_remap import APTOS_CLASSES

CLASS_NAMES = [APTOS_CLASSES[i] for i in range(5)]
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


# ---------------------------------------------------------------------------
# 1. Per-class example grid
# ---------------------------------------------------------------------------

def plot_class_grid(
    df,
    out_path: str,
    n_per_class: int = 4,
    image_size: int = 224,
) -> None:
    """
    Show n_per_class sample images for each of the 5 DR severity grades.
    Applies the preprocessing pipeline so images match what the model sees.
    """
    from src.data.preprocessing import load_and_preprocess

    fig = plt.figure(figsize=(n_per_class * 2.5, 5 * 2.5))
    gs = gridspec.GridSpec(5, n_per_class, figure=fig, hspace=0.05, wspace=0.05)

    for cls in range(5):
        cls_df = df[df["label"] == cls].sample(
            min(n_per_class, len(df[df["label"] == cls])),
            random_state=42,
        ).reset_index(drop=True)

        for col in range(n_per_class):
            ax = fig.add_subplot(gs[cls, col])
            if col < len(cls_df):
                img = load_and_preprocess(cls_df.loc[col, "image_path"], size=image_size)
                ax.imshow(img)
            else:
                ax.set_facecolor("black")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[cls], fontsize=10, labelpad=4)
                ax.yaxis.set_label_position("left")
                ax.yaxis.label.set_visible(True)

    fig.suptitle("Sample Images by DR Severity Grade", fontsize=13, y=1.01)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved class grid       -> {out_path}")


# ---------------------------------------------------------------------------
# 2. Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    out_path: str,
) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(5)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix (row-normalized)", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix -> {out_path}")


# ---------------------------------------------------------------------------
# 3. Per-class F1 bar chart
# ---------------------------------------------------------------------------

def plot_f1_bars(
    labels: np.ndarray,
    preds: np.ndarray,
    out_path: str,
) -> None:
    per_class = f1_score(labels, preds, average=None, zero_division=0, labels=list(range(5)))
    macro = per_class.mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASS_NAMES, per_class, color=PALETTE, edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.axhline(macro, color="black", linestyle="--", linewidth=1.2, label=f"Macro F1={macro:.3f}")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.legend(fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved per-class F1     -> {out_path}")


# ---------------------------------------------------------------------------
# 4. ROC curves (one-vs-rest per class + macro)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    out_path: str,
) -> None:
    n_classes = probs.shape[1]
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    macro_tpr = np.linspace(0, 1, 200)
    all_tpr = []

    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(macro_tpr, fpr, tpr)
        all_tpr.append(interp_tpr)
        ax.plot(fpr, tpr, color=PALETTE[i], lw=1.8,
                label=f"{name} (AUC={roc_auc:.3f})")

    macro_auc = auc(macro_tpr, np.mean(all_tpr, axis=0))
    ax.plot(macro_tpr, np.mean(all_tpr, axis=0), color="black",
            lw=2.2, linestyle="--", label=f"Macro avg (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved ROC curves       -> {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(probs: np.ndarray, labels: np.ndarray, df, out_dir: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = probs.argmax(axis=1)

    if df is not None:
        plot_class_grid(df, str(out_dir / "per_class_grid.png"))
    plot_confusion_matrix(labels, preds, str(out_dir / "confusion_matrix.png"))
    plot_f1_bars(labels, preds, str(out_dir / "per_class_f1.png"))
    plot_roc_curves(labels, probs, str(out_dir / "roc_curves.png"))

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs_npy",  required=True, type=str)
    parser.add_argument("--labels_npy", required=True, type=str)
    parser.add_argument("--data_root",  default=None,  type=str,
                        help="If provided, also generate per-class image grid")
    parser.add_argument("--out_dir",    default="experiments/figures/", type=str)
    parser.add_argument("--config",     default="configs/base.yaml",    type=str)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    probs  = np.load(args.probs_npy)
    labels = np.load(args.labels_npy)

    df = None
    if args.data_root:
        from src.data.build_dataframe import build_train_df
        from src.data.image_audit import audit
        df = audit(build_train_df(args.data_root))

    main(probs, labels, df, args.out_dir)
