"""
Phase 5e — Post-hoc threshold optimization.

Instead of argmax on probabilities, find K-1 thresholds on cumulative
probabilities that directly maximize QWK on the validation set.

Works with any model that outputs softmax probabilities of shape (N, 5).

Usage:
    python -m src.eval.threshold_optimizer \
        --probs_npy path/to/valid_probs.npy \
        --labels_npy path/to/valid_labels.npy
"""

import argparse
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

from src.eval.metrics import compute_metrics, print_metrics


def probs_to_labels_with_thresholds(
    probs: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """
    Map softmax probs (N, 5) → class labels using cumulative probability thresholds.

    cumulative[i, k] = P(y <= k) = sum(probs[i, 0..k])
    label = first k where cumulative >= thresholds[k], else 4.

    thresholds shape: (4,)  — one per class boundary (0|1, 1|2, 2|3, 3|4)
    """
    cum = np.cumsum(probs, axis=1)[:, :4]       # (N, 4): P(y<=0..3)
    preds = np.full(len(probs), 4, dtype=int)
    for k in range(3, -1, -1):                  # iterate high→low so lowest wins
        preds[cum[:, k] >= thresholds[k]] = k
    return preds


def _neg_qwk(thresholds: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.clip(thresholds, 0.01, 0.99)
    # Enforce monotonicity: thresholds must be increasing
    thresholds = np.sort(thresholds)
    preds = probs_to_labels_with_thresholds(probs, thresholds)
    return -cohen_kappa_score(labels, preds, weights="quadratic")


def optimize_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    n_restarts: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Find thresholds that maximize QWK on (probs, labels).
    Returns (best_thresholds, best_qwk).
    """
    rng = np.random.default_rng(seed)
    best_qwk = -1.0
    best_thresholds = np.array([0.2, 0.4, 0.6, 0.8])

    for _ in range(n_restarts):
        init = np.sort(rng.uniform(0.1, 0.9, size=4))
        result = minimize(
            _neg_qwk,
            init,
            args=(probs, labels),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5},
        )
        qwk = -result.fun
        if qwk > best_qwk:
            best_qwk = qwk
            best_thresholds = np.sort(np.clip(result.x, 0.01, 0.99))

    return best_thresholds, best_qwk


def evaluate_with_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    preds = probs_to_labels_with_thresholds(probs, thresholds)
    return compute_metrics(labels, preds, probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs_npy",  required=True, type=str,
                        help="Path to .npy file with softmax probs (N, 5)")
    parser.add_argument("--labels_npy", required=True, type=str,
                        help="Path to .npy file with integer labels (N,)")
    parser.add_argument("--n_restarts", default=10, type=int)
    args = parser.parse_args()

    probs  = np.load(args.probs_npy)
    labels = np.load(args.labels_npy)

    # Baseline (argmax)
    baseline_preds = probs.argmax(axis=1)
    baseline_m = compute_metrics(labels, baseline_preds, probs)
    print_metrics(baseline_m, prefix="baseline (argmax)")

    # Optimized thresholds
    thresholds, best_qwk = optimize_thresholds(probs, labels, n_restarts=args.n_restarts)
    optimized_m = evaluate_with_thresholds(probs, labels, thresholds)

    print(f"\nOptimized thresholds: {np.round(thresholds, 4)}")
    print_metrics(optimized_m, prefix="optimized (thresholds)")
    print(f"QWK improvement: {optimized_m['qwk'] - baseline_m['qwk']:+.4f}")
