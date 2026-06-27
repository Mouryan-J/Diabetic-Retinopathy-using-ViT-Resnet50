"""
Gradio inference app for Diabetic Retinopathy grading.

Loads the best ViT checkpoint from Hugging Face Hub (or a local path).
Preprocessing matches the training pipeline exactly:
  circular fundus crop -> CLAHE green channel -> resize -> normalize

Usage:
    python app.py
    python app.py --checkpoint checkpoints/vit/best_phaseC.pt  # local
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image

from src.data.preprocessing import preprocess
from src.eval.label_remap import APTOS_CLASSES, remap

# ---- run label remap tests before serving ----
from src.eval import label_remap
label_remap.test_remap_is_correct()
label_remap.test_remap_preserves_qwk_sign()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_REPO  = "Mouryan-J/dr-vit-resnet"          # update after pushing to Hub
HF_FILE  = "best_phaseC.pt"
IMG_SIZE = 384
NUM_CLASSES = 5
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES  = [APTOS_CLASSES[i] for i in range(NUM_CLASSES)]
SEVERITY_CSS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
SEVERITY_EMO = ["✅", "🟡", "🟠", "🔴", "🟣"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str | None):
    from src.models.vit_model import build_vit
    model = build_vit(num_classes=NUM_CLASSES).to(DEVICE)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded local checkpoint: {checkpoint_path}")
    else:
        # Download from Hugging Face Hub
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
            ckpt  = torch.load(local, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint from Hub: {HF_REPO}/{HF_FILE}")
        except Exception as e:
            print(f"WARNING: could not load weights ({e}). Running with random weights.")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB → preprocessed normalized tensor (1, 3, H, W)."""
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr = preprocess(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img


def predict(pil_img: Image.Image) -> tuple[dict, str]:
    """
    Returns:
        confidences : dict {class_name: probability} for Gradio Label component
        summary     : markdown string with predicted grade + description
    """
    if pil_img is None:
        return {}, "Please upload a fundus image."

    tensor = _preprocess_pil(pil_img)
    with torch.no_grad():
        logits = MODEL(tensor).logits
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    raw_pred  = int(probs.argmax())
    aptos_pred = int(remap(np.array([raw_pred]))[0])
    confidence = float(probs[raw_pred])

    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    emo = SEVERITY_EMO[aptos_pred]
    summary = (
        f"### {emo} Predicted Grade: **{CLASS_NAMES[aptos_pred]}** "
        f"(Grade {aptos_pred})\n\n"
        f"**Confidence:** {confidence:.1%}\n\n"
        f"*This tool is for research/educational purposes only — "
        f"not a clinical diagnostic instrument.*"
    )

    return confidences, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Diabetic Retinopathy Grader", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Diabetic Retinopathy Severity Grader\n"
            "Upload a fundus photograph to predict the DR severity grade (0–4).\n"
            "Model: **ViT-Base/16-384** fine-tuned on APTOS 2019."
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Fundus Image")
                submit    = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                label_out   = gr.Label(num_top_classes=5, label="Class Probabilities")
                summary_out = gr.Markdown()

        submit.click(fn=predict, inputs=img_input, outputs=[label_out, summary_out])

        gr.Examples(
            examples=[],   # add example image paths here after training
            inputs=img_input,
        )

        gr.Markdown(
            "---\n"
            "**Grades:** 0=No DR | 1=Mild | 2=Moderate | 3=Severe | 4=Proliferative DR\n\n"
            "**Source:** [GitHub](https://github.com/Mouryan-J/Diabetic-Retinopathy-using-ViT-Resnet50) · "
            "**Weights:** Hugging Face Hub"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Local .pt checkpoint path (skips Hub download)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    MODEL = _load_model(args.checkpoint)
    demo  = build_ui()
    demo.launch(share=args.share)
