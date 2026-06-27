"""
Phase 5a — Ordinal regression losses for DR severity (ordered 0-4 scale).

OrdinalCrossEntropyLoss:
    Encodes each label as K-1 cumulative binary targets [p(y>0), p(y>1), ...]
    and uses BCE loss. Requires the model to output K-1 raw logits (not softmax).

    Reference: Frank & Hall (2001) / Cheng et al. (2008).

CORNLoss (Conditional Ordinal Regression for Neural Networks):
    Trains K-1 binary classifiers where each predicts p(y > k | y > k-1).
    Reference: Shi et al. (2021) — https://arxiv.org/abs/2111.08851

Usage with ViT — swap the head and loss in train_vit.py:
    model.classifier = nn.Linear(hidden_size, num_classes - 1)
    criterion = OrdinalCrossEntropyLoss() or CORNLoss(num_classes)
    preds = ordinal_logits_to_label(logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# label encoders
# ---------------------------------------------------------------------------

def label_to_cumulative(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer class labels to cumulative binary targets.
    label k  →  [1, 1, ..., 1, 0, 0, ..., 0]  (k ones, then K-1-k zeros)
    Shape: (N,) → (N, num_classes-1)
    """
    K = num_classes - 1
    targets = torch.zeros(labels.size(0), K, device=labels.device, dtype=torch.float32)
    for i in range(K):
        targets[:, i] = (labels > i).float()
    return targets


def ordinal_logits_to_label(logits: torch.Tensor) -> torch.Tensor:
    """Convert K-1 cumulative logits → class index (0…K)."""
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1).long()


def ordinal_logits_to_probs(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert K-1 cumulative logits → class probability distribution (N, K).
    P(y=k) = P(y>k-1) - P(y>k)   with  P(y>-1)=1, P(y>K-1)=0.
    """
    cum = torch.sigmoid(logits)                          # (N, K-1)
    ones  = torch.ones(cum.size(0), 1, device=logits.device)
    zeros = torch.zeros(cum.size(0), 1, device=logits.device)
    cum_ext = torch.cat([ones, cum, zeros], dim=1)       # (N, K+1)
    probs = cum_ext[:, :-1] - cum_ext[:, 1:]             # (N, K)
    return probs.clamp(min=0.0)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

class OrdinalCrossEntropyLoss(nn.Module):
    """Binary cross-entropy on cumulative targets (expects K-1 output logits)."""

    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1) + 1
        targets = label_to_cumulative(labels, num_classes)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.class_weights is not None:
            # weight each sample by its class weight
            w = self.class_weights[labels].unsqueeze(1)
            loss = loss * w
        return loss.mean()


class CORNLoss(nn.Module):
    """
    CORN loss: each rank-k classifier trained only on samples with label >= k.
    Expects K-1 output logits.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        K = self.num_classes - 1
        loss = torch.tensor(0.0, device=logits.device)
        n_terms = 0
        for k in range(K):
            mask = labels >= k
            if mask.sum() == 0:
                continue
            logit_k = logits[mask, k]
            target_k = (labels[mask] > k).float()
            loss = loss + F.binary_cross_entropy_with_logits(logit_k, target_k)
            n_terms += 1
        return loss / max(n_terms, 1)
