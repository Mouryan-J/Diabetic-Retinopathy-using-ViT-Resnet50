#!/usr/bin/env bash
# Run this once at the top of every new Colab session.
# Usage (in a Colab cell):
#   !bash scripts/colab_setup.sh

set -euo pipefail

echo "=== Installing dependencies ==="
pip install -q \
    torch torchvision \
    transformers==4.41.2 \
    "timm>=0.9.16" \
    albumentations==1.4.10 \
    scikit-learn pandas numpy \
    opencv-python-headless \
    gradio kaggle matplotlib seaborn tqdm pyyaml scipy

echo "=== Checking GPU ==="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
"

echo "=== Setup complete ==="
