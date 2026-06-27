"""
Compute inverse-frequency class weights for weighted cross-entropy.
Returns a float32 tensor of shape (num_classes,) on the requested device.
"""

import numpy as np
import torch
import pandas as pd


def compute_class_weights(
    labels: np.ndarray | pd.Series,
    num_classes: int = 5,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    # Inverse frequency, normalized so weights sum to num_classes
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.sum() * num_classes  # keep scale reasonable
    return torch.tensor(weights, dtype=torch.float32, device=device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/", type=str)
    args = parser.parse_args()

    from src.data.build_dataframe import build_train_df
    from src.data.image_audit import audit

    df = audit(build_train_df(args.data_root))
    w = compute_class_weights(df["label"])
    for i, wi in enumerate(w):
        print(f"  Class {i}: {wi:.4f}")
