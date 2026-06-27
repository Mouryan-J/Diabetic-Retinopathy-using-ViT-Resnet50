"""
Evaluation metrics for 5-class DR grading.
All functions accept plain Python lists or numpy arrays.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None = None,
    num_classes: int = 5,
) -> dict:
    """
    Args:
        labels : ground-truth class indices (N,)
        preds  : predicted class indices (N,)
        probs  : softmax probabilities (N, C) — required for AUC, optional otherwise
    Returns dict with keys: accuracy, f1_macro, qwk, auc_macro
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")

    auc_macro = float("nan")
    if probs is not None:
        probs = np.asarray(probs)
        try:
            auc_macro = roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            pass  # happens if a class has no positive samples in the batch

    return {
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "qwk": round(float(qwk), 4),
        "auc_macro": round(float(auc_macro), 4),
    }


def print_metrics(metrics: dict, prefix: str = "") -> None:
    tag = f"[{prefix}] " if prefix else ""
    print(
        f"{tag}Accuracy={metrics['accuracy']:.4f}  "
        f"F1={metrics['f1_macro']:.4f}  "
        f"QWK={metrics['qwk']:.4f}  "
        f"AUC={metrics['auc_macro']:.4f}"
    )
