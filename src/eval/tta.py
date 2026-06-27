"""
Phase 5b — Test-Time Augmentation (TTA) for ViT (PyTorch).

Runs N augmented views of each image and averages the softmax probabilities.
Augmentations used: original + horizontal flip + vertical flip + 90/180/270 rotations.

Usage:
    python -m src.eval.tta \
        --checkpoint checkpoints/vit/best_phaseC.pt \
        --config configs/base.yaml \
        --split valid
"""

import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.build_dataframe import build_train_df
from src.data.image_audit import audit
from src.data.split import stratified_split
from src.data.dataset import RetinopathyDataset
from src.data.augmentations import valid_transforms
from src.models.vit_model import build_vit
from src.eval.metrics import compute_metrics, print_metrics

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _tta_transforms(image_size: int) -> list[A.Compose]:
    base = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)]   + base),
        A.Compose([A.Rotate(limit=(90, 90), p=1.0)]  + base),
        A.Compose([A.Rotate(limit=(180, 180), p=1.0)] + base),
        A.Compose([A.Rotate(limit=(270, 270), p=1.0)] + base),
    ]


def tta_predict(
    model: torch.nn.Module,
    df,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return averaged softmax probabilities (N, num_classes) over all TTA views."""
    transforms_list = _tta_transforms(image_size)
    all_probs = []

    for tfm in transforms_list:
        ds = RetinopathyDataset(df, image_size=image_size, transform=tfm, has_labels=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

        view_probs = []
        model.eval()
        with torch.no_grad():
            for imgs in tqdm(loader, desc="TTA view", leave=False):
                imgs = imgs.to(device)
                logits = model(imgs).logits
                probs = F.softmax(logits, dim=1).cpu().numpy()
                view_probs.append(probs)
        all_probs.append(np.vstack(view_probs))

    return np.mean(all_probs, axis=0)


def main(cfg: dict, checkpoint: str, split: str = "valid") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = audit(build_train_df(cfg["data"]["root"]))
    train_df, valid_df = stratified_split(
        df,
        valid_fraction=cfg["data"]["valid_fraction"],
        random_seed=cfg["data"]["random_seed"],
    )
    eval_df = valid_df if split == "valid" else train_df

    model = build_vit(num_classes=cfg["data"]["num_classes"]).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    probs = tta_predict(
        model, eval_df,
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        device=device,
    )
    preds = probs.argmax(axis=1)
    labels = eval_df["label"].values
    m = compute_metrics(labels, preds, probs)
    print("\n=== TTA Results ===")
    print_metrics(m, prefix="TTA")
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", default="configs/base.yaml", type=str)
    parser.add_argument("--split", default="valid", choices=["valid", "train"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.checkpoint, args.split)
