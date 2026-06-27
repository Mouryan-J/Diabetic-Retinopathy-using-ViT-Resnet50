#!/usr/bin/env bash
# Download APTOS 2019 Blindness Detection data via Kaggle CLI.
# Prerequisites: `pip install kaggle` and ~/.kaggle/kaggle.json with your API key.
# Usage: bash scripts/download_data.sh

set -euo pipefail

COMPETITION="aptos2019-blindness-detection"
DATA_DIR="data"

echo "Downloading competition data: $COMPETITION"
mkdir -p "$DATA_DIR"

kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

echo "Unzipping..."
unzip -q "$DATA_DIR/${COMPETITION}.zip" -d "$DATA_DIR"
rm "$DATA_DIR/${COMPETITION}.zip"

# The competition zip contains train_images/, test_images/, train.csv, test.csv
echo "Done. Contents of $DATA_DIR:"
ls "$DATA_DIR"
