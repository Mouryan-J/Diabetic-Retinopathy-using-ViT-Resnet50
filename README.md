# Diabetic Retinopathy Grading — ViT + ResNet-50

5-class severity grading of fundus photographs using the
[APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset.

**Live demo:** coming after training · **Weights:** Hugging Face Hub (link TBD)  
**Repo:** https://github.com/Mouryan-J/Diabetic-Retinopathy-using-ViT-Resnet50

---

## Results

| Model | Accuracy | F1 Macro | QWK | AUC Macro |
|-------|----------|----------|-----|-----------|
| Baseline (old notebook, ResNet-50) | 0.825 | 0.56 | 0.887 | 0.59 |
| ViT-Base/16-384 (Phase 3) | — | — | — | — |
| ResNet-50 PyTorch (Phase 4) | — | — | — | — |
| Best ensemble + thresholds | — | — | — | — |

*Table updated after training completes.*

---

## Architecture

```
Fundus image
     │
     ▼
[Circular crop]──removes black border padding
     │
     ▼
[CLAHE on green channel]──local contrast enhancement
     │
     ▼
[Resize 384×384]
     │
     ├──► ViT-Base/16-384 ──3-phase fine-tune──► logits (5)
     │         HuggingFace Transformers
     │
     └──► ResNet-50 ──────2-phase fine-tune──► logits (5)
               timm (ImageNet pretrained)
                    │
                    ▼
          [Weighted average ensemble]
                    │
                    ▼
          [Threshold optimization]
                    │
                    ▼
            DR Grade  0–4
```

---

## Project Structure

```
.
├── app.py                        # Gradio inference app
├── configs/
│   ├── base.yaml                 # Default hyperparameters
│   └── colab.yaml                # Google Colab overrides
├── experiments/
│   └── results.md                # Experiment log table
├── scripts/
│   ├── colab_setup.sh            # Colab dependency installer
│   ├── download_data.sh          # Linux/Mac data download
│   └── download_data.ps1         # Windows PowerShell data download
├── src/
│   ├── data/
│   │   ├── build_dataframe.py    # CSV → DataFrame
│   │   ├── image_audit.py        # Drop corrupt images
│   │   ├── split.py              # Stratified train/valid split
│   │   ├── preprocessing.py      # Fundus crop + CLAHE
│   │   ├── augmentations.py      # Albumentations pipelines
│   │   └── dataset.py            # PyTorch Dataset
│   ├── models/
│   │   ├── vit_model.py          # ViT-Base/16-384 (HF)
│   │   ├── resnet_model.py       # ResNet-50 (timm)
│   │   └── convnext_model.py     # ConvNeXt-Base (timm)
│   ├── training/
│   │   ├── class_weights.py      # Inverse-frequency weights
│   │   ├── ordinal_loss.py       # CORN + ordinal BCE losses
│   │   ├── train_vit.py          # ViT 3-phase trainer
│   │   ├── train_resnet.py       # ResNet 3-phase trainer
│   │   ├── train_convnext.py     # ConvNeXt 3-phase trainer
│   │   └── kfold_train.py        # 5-fold CV ensemble
│   └── eval/
│       ├── metrics.py            # Accuracy, F1, QWK, AUC
│       ├── label_remap.py        # APTOS label mapping + unit tests
│       ├── evaluate_test.py      # Full test-set evaluation
│       ├── tta.py                # Test-time augmentation
│       ├── threshold_optimizer.py# Post-hoc QWK threshold tuning
│       ├── ensemble.py           # Multi-model averaging
│       └── visualize.py          # Plots: grid, CM, F1, ROC
├── MODEL_CARD.md
└── requirements.txt
```

---

## Quickstart — Google Colab

```python
# Cell 1 — mount Drive (checkpoints persist here)
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — clone repo and install deps
!git clone https://github.com/Mouryan-J/Diabetic-Retinopathy-using-ViT-Resnet50.git
%cd Diabetic-Retinopathy-using-ViT-Resnet50
!bash scripts/colab_setup.sh

# Cell 3 — download APTOS data (needs kaggle.json in Drive)
import shutil, os
shutil.copy('/content/drive/MyDrive/kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
!bash scripts/download_data.sh

# Cell 4 — train ViT
!python -m src.training.train_vit --config configs/colab.yaml

# Cell 5 — train ResNet
!python -m src.training.train_resnet --config configs/colab.yaml

# Cell 6 — evaluate
!python -m src.eval.evaluate_test \
    --config configs/colab.yaml \
    --checkpoint /content/drive/MyDrive/DR_checkpoints/vit/best_phaseC.pt \
    --use_valid --out_dir experiments/

# Cell 7 — visualizations
!python -m src.eval.visualize \
    --probs_npy experiments/test_probs_vit.npy \
    --labels_npy data/valid_labels.npy \
    --data_root data/ --out_dir experiments/figures/
```

---

## Quickstart — Local (Windows, RTX GPU)

```powershell
# Install deps (PyTorch must be installed first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Download data (needs ~/.kaggle/kaggle.json)
.\scripts\download_data.ps1

# Train
python -m src.training.train_vit --config configs/base.yaml
python -m src.training.train_resnet --config configs/base.yaml

# Evaluate
python -m src.eval.evaluate_test `
    --checkpoint checkpoints/vit/best_phaseC.pt `
    --use_valid --out_dir experiments/
```

---

## Preprocessing Details

**Circular fundus crop** — OpenCV contour detection finds the largest circle
in the grayscale image, masks pixels outside it, then crops to the bounding box.
Eliminates uninformative black border padding present in most APTOS images.

**CLAHE on green channel** — Contrast Limited Adaptive Histogram Equalization
applied only to the green channel (most discriminative for DR lesions).
`clipLimit=2.0`, `tileGridSize=8×8`. Significantly improves visibility of
micro-aneurysms, haemorrhages, and hard exudates.

---

## Training Details

### ViT — 3-phase fine-tuning

| Phase | Unfrozen layers | LR | Epochs | Scheduler |
|-------|----------------|-----|--------|-----------|
| A | Classifier head only | 1e-3 | 5 | — |
| B | Transformer blocks 6-11 | 5e-5 | 10 | Cosine |
| C | All layers | 1e-5 | 10 | Cosine + early stop |

- Loss: weighted cross-entropy + label smoothing (0.1)
- Optimizer: AdamW (weight_decay=1e-4)
- AMP (mixed precision) enabled

### ResNet-50 — 3-phase fine-tuning

| Phase | Unfrozen layers | LR | Epochs |
|-------|----------------|-----|--------|
| 1 | fc head only | 1e-3 | 5 |
| 2 | layer3/4 + fc | 1e-4 | 7 |
| 3 | All layers | 1e-5 | 8 |

---

## QWK Improvement Experiments

| Experiment | Script | Expected gain |
|------------|--------|--------------|
| 5a. Ordinal loss (CORN) | `src/training/ordinal_loss.py` | Aligns loss with QWK metric |
| 5b. TTA (6 views) | `src/eval/tta.py` | ~+0.01–0.02 QWK |
| 5c. 5-fold ensemble | `src/training/kfold_train.py` | Variance reduction |
| 5d. ConvNeXt-B ensemble | `src/training/train_convnext.py` | Architecture diversity |
| 5e. Threshold optimization | `src/eval/threshold_optimizer.py` | Direct QWK maximization |

---

## Label Safety

The original notebook briefly produced negative QWK due to a label-order
inversion bug. This repo has an explicit `TRAIN_TO_APTOS` mapping in
[src/eval/label_remap.py](src/eval/label_remap.py) with 3 unit tests —
including a QWK sign check — that run automatically before every evaluation.

```bash
python -m pytest src/eval/label_remap.py -v
```

---

## Citation / Dataset

```
@misc{aptos2019,
  title  = {APTOS 2019 Blindness Detection},
  author = {Asia Pacific Tele-Ophthalmology Society},
  year   = {2019},
  url    = {https://www.kaggle.com/c/aptos2019-blindness-detection}
}
```
