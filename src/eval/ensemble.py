"""
Multi-model ensemble utility.

Loads probability arrays from multiple sources (ViT, ResNet, ConvNeXt,
k-fold OOF files) and averages them.  Also supports weighted averaging.

Usage:
    python -m src.eval.ensemble \
        --probs checkpoints/vit/valid_probs.npy \
                checkpoints/resnet/valid_probs.npy \
        --labels data/valid_labels.npy \
        --weights 0.6 0.4
"""

import argparse
import numpy as np

from src.eval.metrics import compute_metrics, print_metrics
from src.eval.threshold_optimizer import optimize_thresholds, evaluate_with_thresholds


def ensemble_probs(
    prob_arrays: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """
    Weighted average of softmax probability arrays.
    Each array: (N, num_classes).
    """
    if weights is None:
        weights = [1.0] * len(prob_arrays)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    result = np.zeros_like(prob_arrays[0])
    for prob, w in zip(prob_arrays, weights):
        result += w * prob
    return result


def load_kfold_oof(ckpt_dir: str, n_folds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate OOF probs/labels saved by kfold_train.py."""
    from pathlib import Path
    ckpt_dir = Path(ckpt_dir)
    probs_list, labels_list = [], []
    for fold in range(1, n_folds + 1):
        probs_list.append(np.load(ckpt_dir / f"fold{fold}_oof_probs.npy"))
        labels_list.append(np.load(ckpt_dir / f"fold{fold}_oof_labels.npy"))
    return np.vstack(probs_list), np.concatenate(labels_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs", nargs="+", required=True, type=str,
                        help=".npy files with softmax probs (N, 5) — one per model")
    parser.add_argument("--labels", required=True, type=str,
                        help=".npy file with integer labels (N,)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional per-model weights (default: equal)")
    parser.add_argument("--optimize_thresholds", action="store_true",
                        help="Run post-hoc threshold optimization on ensemble output")
    args = parser.parse_args()

    prob_arrays = [np.load(p) for p in args.probs]
    labels = np.load(args.labels)

    # Individual model baselines
    for path, probs in zip(args.probs, prob_arrays):
        m = compute_metrics(labels, probs.argmax(axis=1), probs)
        print_metrics(m, prefix=path.split("/")[-1])

    # Ensemble
    ens_probs = ensemble_probs(prob_arrays, weights=args.weights)
    ens_m = compute_metrics(labels, ens_probs.argmax(axis=1), ens_probs)
    print("\n=== Ensemble ===")
    print_metrics(ens_m, prefix="ensemble")

    if args.optimize_thresholds:
        thresholds, _ = optimize_thresholds(ens_probs, labels)
        opt_m = evaluate_with_thresholds(ens_probs, labels, thresholds)
        print(f"\nOptimized thresholds: {np.round(thresholds, 4)}")
        print_metrics(opt_m, prefix="ensemble + thresholds")
