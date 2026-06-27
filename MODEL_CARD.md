# Model Card — Diabetic Retinopathy Grading (ViT + ResNet-50)

## Task
5-class severity grading of fundus photographs for Diabetic Retinopathy (DR)
using the APTOS 2019 Blindness Detection dataset.

**Severity scale:**
| Label | Grade |
|-------|-------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## Dataset
- **Source:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- **Train:** 3,662 fundus images
- **Validation split:** 20% stratified hold-out (random seed 42)
- **Test:** Official Kaggle test set

---

## Metrics Summary

| Experiment | Model | Accuracy | F1 Macro | QWK | AUC Macro |
|------------|-------|----------|----------|-----|-----------|
| Baseline (old notebook) | ResNet-50 (TF) | 0.825 | 0.56 | 0.887 | 0.59 |
| Phase 3 | ViT-B/16-384 (3-phase) | — | — | — | — |
| Phase 4 | ResNet-50 (2-phase) | — | — | — | — |
| Phase 5a | ViT + Ordinal Loss | — | — | — | — |
| Phase 5b | ViT + TTA (6 views) | — | — | — | — |
| Phase 5c | ViT × 5-fold Ensemble | — | — | — | — |
| Phase 5d | ConvNeXt-B ensemble | — | — | — | — |
| Phase 5e | Ensemble + Threshold Opt | — | — | — | — |

*Blank cells to be filled after training runs complete.*

---

## Architecture

### ViT (Primary Model)
- **Backbone:** `google/vit-base-patch16-384` (HuggingFace Transformers)
- **Input resolution:** 384 × 384 RGB
- **Head:** Linear(768 → 5)
- **3-phase fine-tuning:**
  - Phase A: classifier head only (5 epochs, lr=1e-3)
  - Phase B: blocks 6-11 unfrozen (10 epochs, lr=5e-5, cosine decay)
  - Phase C: full unfreeze (10 epochs, lr=1e-5, cosine decay, early stop on QWK)
- **Loss:** Weighted cross-entropy + label smoothing (0.1)

### ResNet-50 (Secondary Model)
- **Backbone:** `ResNet50` (Keras, ImageNet weights)
- **Input resolution:** 384 × 384 RGB
- **Head:** GlobalAvgPool → Dropout(0.3) → Dense(5, softmax)
- **2-phase fine-tuning:**
  - Phase 1: head only (5 epochs, lr=1e-3)
  - Phase 2: full unfreeze (15 epochs, lr=1e-5, ReduceLROnPlateau)

### ConvNeXt-Base (Ensemble Member)
- **Backbone:** `convnext_base.fb_in22k_ft_in1k_384` (timm, IN-22k pretrain)
- **Input resolution:** 384 × 384 RGB
- Same 3-phase freeze schedule as ViT

---

## Preprocessing Pipeline
Every image goes through this pipeline at load time, before augmentation:

1. **Circular fundus crop** — contour-based detection removes black border padding
2. **CLAHE on green channel** — improves local contrast of retinal lesions (clipLimit=2.0, tileGrid=8×8)
3. **Resize** to 384 × 384
4. **Normalize** with ImageNet mean/std

---

## Training Augmentations (train split only)
- RandomResizedCrop (scale 0.8-1.0)
- HorizontalFlip, VerticalFlip, Rotate ±20°
- RandomBrightnessContrast / HueSaturationValue
- GaussNoise, GaussianBlur
- CoarseDropout (8 holes, 24×24 px)

---

## QWK Improvement Methods

### What the baseline did
The original notebook trained a ResNet-50 with standard cross-entropy and
plain argmax decoding, achieving QWK=0.887. It suffered a brief negative-QWK
episode caused by a label-order inversion bug (since fixed and unit-tested).

### What we changed and why each should improve QWK

| Change | Expected benefit |
|--------|-----------------|
| Fundus crop + CLAHE preprocessing | Removes uninformative black borders; CLAHE highlights micro-aneurysms and haemorrhages that discriminate adjacent severity grades |
| 3-phase ViT fine-tuning | ViT's global self-attention captures long-range retinal structure that CNNs miss; gradual unfreezing prevents catastrophic forgetting of ImageNet features |
| Weighted cross-entropy | APTOS is heavily class-imbalanced (No DR >> Proliferative); weighting prevents collapse onto majority class |
| Label smoothing (0.1) | Reduces overconfidence on adjacent ordinal classes where inter-grader disagreement exists |
| Ordinal loss (Phase 5a) | Cross-entropy treats all misclassifications equally; ordinal loss penalises grade-2→grade-0 errors more than grade-2→grade-1, directly aligning training with QWK |
| TTA × 6 views (Phase 5b) | Averages out augmentation noise at inference; reliable +0.01-0.02 QWK improvement with no extra training |
| 5-fold ensemble (Phase 5c) | Reduces variance; OOF averaging is particularly effective when training set is small (~3k images) |
| ConvNeXt-B ensemble (Phase 5d) | Architecture diversity — CNN inductive bias complements ViT's attention mechanism |
| Threshold optimization (Phase 5e) | argmax is not optimal for QWK; tuning cumulative probability thresholds directly on the validation set targets the metric |

---

## Label Remap
Training labels (APTOS `diagnosis` column) and evaluation labels use the
same 0-4 integer convention.  An explicit `TRAIN_TO_APTOS` mapping in
`src/eval/label_remap.py` is unit-tested (3 tests including a QWK sign check)
and applied before every evaluation run.

---

## Reproducibility
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
bash scripts/download_data.sh

# 3. Train ViT
python -m src.training.train_vit --config configs/base.yaml

# 4. Train ResNet
python -m src.training.train_resnet --config configs/base.yaml

# 5. Evaluate with label remap tests
python -m src.eval.evaluate_test \
    --checkpoint checkpoints/vit/best_phaseC.pt \
    --use_valid --out_dir experiments/

# 6. Generate figures
python -m src.eval.visualize \
    --probs_npy experiments/test_probs_vit.npy \
    --labels_npy data/valid_labels.npy \
    --data_root data/ --out_dir experiments/figures/
```

---

## Weights
Model weights are hosted on Hugging Face Hub (not in this repository — file
size limits). See `app.py` for the Hub repo path and loading code.

---

## Intended Use & Limitations
- **Intended use:** Research / educational demonstration of DR grading.
- **Not validated** for clinical diagnosis or screening.
- Performance may degrade on fundus images from cameras/populations not
  represented in APTOS 2019.
- Severity grades 1 (Mild) and 3 (Severe) are typically the hardest to
  classify due to small training set size and inter-grader variability.
