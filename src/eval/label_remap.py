"""
Phase 6 — Explicit label remapping between training order and APTOS order.

APTOS 2019 uses this grading scale (official):
  0 = No DR
  1 = Mild
  2 = Moderate
  3 = Severe
  4 = Proliferative DR

Our training labels (from train.csv 'diagnosis' column) use the same 0-4
integers in the same order, so NO remapping is needed for the APTOS dataset.

This module makes that identity mapping explicit and testable so that:
  (a) future dataset swaps must update remap() and the unit test together, and
  (b) the negative-QWK bug from the original notebook (caused by a silent
      label inversion) cannot silently reappear.

Run unit tests:
    python -m pytest src/eval/label_remap.py -v
    python -m src.eval.label_remap          (standalone smoke test)
"""

import numpy as np

# Official APTOS severity names indexed by label integer
APTOS_CLASSES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

# Mapping from our training label → APTOS evaluation label.
# Values: TRAIN_LABEL -> APTOS_LABEL
# Currently identity (same dataset, same convention).
TRAIN_TO_APTOS: dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}


def remap(labels: np.ndarray) -> np.ndarray:
    """Apply TRAIN_TO_APTOS mapping to an array of predicted labels."""
    labels = np.asarray(labels, dtype=int)
    mapping = np.array([TRAIN_TO_APTOS[k] for k in range(len(TRAIN_TO_APTOS))], dtype=int)
    return mapping[labels]


def label_name(aptos_label: int) -> str:
    return APTOS_CLASSES[aptos_label]


# ---------------------------------------------------------------------------
# Unit tests (run via pytest or directly)
# ---------------------------------------------------------------------------

def test_remap_is_correct():
    """Remap must be a valid permutation of 0-4 matching APTOS convention."""
    for train_lbl, aptos_lbl in TRAIN_TO_APTOS.items():
        assert 0 <= train_lbl <= 4, f"Invalid training label: {train_lbl}"
        assert 0 <= aptos_lbl <= 4, f"Invalid APTOS label: {aptos_lbl}"

    # Must be a bijection (no two training labels map to the same APTOS label)
    aptos_values = list(TRAIN_TO_APTOS.values())
    assert len(set(aptos_values)) == len(aptos_values), \
        "TRAIN_TO_APTOS is not a bijection — duplicate APTOS labels detected"

    # All 5 labels must be covered
    assert set(TRAIN_TO_APTOS.keys()) == {0, 1, 2, 3, 4}
    assert set(aptos_values) == {0, 1, 2, 3, 4}


def test_remap_vectorized():
    """Vectorized remap must match element-wise application."""
    labels = np.array([0, 1, 2, 3, 4, 2, 0])
    expected = np.array([TRAIN_TO_APTOS[l] for l in labels])
    np.testing.assert_array_equal(remap(labels), expected)


def test_remap_preserves_qwk_sign():
    """
    After correct remapping, QWK on perfect predictions must be 1.0.
    This catches the sign-flip bug (negative QWK on correct predictions).
    """
    from sklearn.metrics import cohen_kappa_score
    labels = np.array([0, 1, 2, 3, 4] * 20)
    remapped_preds = remap(labels)   # identity remap → same as labels
    qwk = cohen_kappa_score(labels, remapped_preds, weights="quadratic")
    assert qwk == 1.0, f"QWK should be 1.0 on perfect predictions, got {qwk}"


if __name__ == "__main__":
    test_remap_is_correct()
    test_remap_vectorized()
    test_remap_preserves_qwk_sign()
    print("All label_remap tests passed.")
    print("TRAIN_TO_APTOS mapping:")
    for k, v in TRAIN_TO_APTOS.items():
        print(f"  train {k} ({label_name(k)}) -> aptos {v} ({label_name(v)})")
