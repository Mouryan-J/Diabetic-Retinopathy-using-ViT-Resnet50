import numpy as np
from PIL import Image

import gradio as gr

# --- TF & Torch stacks ---
import tensorflow as tf
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


# =========================================================
# 0) Configuration
# =========================================================
# Where your models live in the Space
RESNET_PATH = "retinopathy_baseline_model.keras"
VIT_DIR     = "dr-vit-EXPORT-final"

# APTOS display order (we always return in this order)
APTOS_LABELS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

# Your training folder order for both models was alphabetical:
# ['Mild','Moderate','No_DR','Proliferate_DR','Severe']
# Convert FROM that order TO APTOS order: [2,0,1,4,3]
TRAINING_ORDER = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]
IDX_MAP_TRAINING_TO_APTOS = [2, 0, 1, 4, 3]

# Tell the app how your ResNet was exported.
#   "has_rescaling"     -> model already includes layers.Rescaling(1/255)
#   "manual_divide_255" -> you need to divide inputs by 255 here
#   "keras_app_resnet50"-> you trained with keras.applications.resnet50.preprocess_input
RESNET_INPUT_MODE = "has_rescaling"   # change if needed: "manual_divide_255" or "keras_app_resnet50"


# =========================================================
# 1) Utilities (softmax + remap + robust class-order handling)
# =========================================================
def ensure_softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax iff x doesn't already look like probabilities."""
    x = np.asarray(x, dtype=np.float32)
    s = float(np.sum(x))
    if not (0.98 <= s <= 1.02) or np.any(x < 0) or np.any(x > 1):
        e = np.exp(x - np.max(x))
        x = e / np.sum(e)
    return x

def remap_probs_to_aptos(probs: np.ndarray, idx_map: list[int]) -> np.ndarray:
    """Return probs in APTOS order given idx_map = [model_idx for aptos_idx]."""
    probs = np.asarray(probs, dtype=np.float32)
    return probs[idx_map]

def normalize_label(s: str) -> str:
    s = str(s).strip().lower()
    if s in {"proliferate_dr", "proliferative_dr"}:
        return "Proliferative_DR"
    if s == "no_dr":
        return "No_DR"
    if s == "mild":
        return "Mild"
    if s == "moderate":
        return "Moderate"
    if s == "severe":
        return "Severe"
    # fallback: keep original (capitalization)
    return s

def build_index_remap(model_order, target_order=APTOS_LABELS):
    """
    model_order: list of class names as used by the model (index 0..C-1)
    target_order: desired order (APTOS)
    returns idx_map such that out_probs[tgt_idx] = in_probs[idx_map[tgt_idx]]
    """
    model_norm  = [normalize_label(s) for s in model_order]
    target_norm = [normalize_label(s) for s in target_order]
    idx_map = []
    for tgt in target_norm:
        if tgt not in model_norm:
            raise ValueError(f"Class '{tgt}' not found in model classes: {model_order}")
        idx_map.append(model_norm.index(tgt))
    return idx_map


# =========================================================
# 2) Load both models
# =========================================================
# --- ResNet (TensorFlow/Keras) ---
resnet_model = tf.keras.models.load_model(RESNET_PATH)

# --- ViT (PyTorch/HF) ---
vit_processor = AutoImageProcessor.from_pretrained(VIT_DIR)
vit_model = AutoModelForImageClassification.from_pretrained(VIT_DIR)
vit_model.eval()  # inference mode (CPU on Spaces)

# Determine ViT class order from config if available; else use your training order
if hasattr(vit_model.config, "id2label") and vit_model.config.id2label:
    # Sort keys numerically to make 0..C-1
    vit_order = [vit_model.config.id2label[i] for i in sorted(vit_model.config.id2label.keys())]
else:
    vit_order = TRAINING_ORDER[:]  # fallback

# Build remaps (defensive)
RESNET_IDX_MAP = build_index_remap(TRAINING_ORDER, APTOS_LABELS)           # [2,0,1,4,3]
VIT_IDX_MAP    = build_index_remap(vit_order, APTOS_LABELS)                # most likely [2,0,1,4,3]


# =========================================================
# 3) Prediction
# =========================================================
def predict(model_choice: str, image):
    """
    Returns probabilities in APTOS order for the chosen model.
    - model_choice: "ViT (PyTorch)" or "ResNet-50 (TensorFlow)"
    - image: PIL.Image (from Gradio)
    """
    if image is None:
        return {label: 0.0 for label in APTOS_LABELS}

    # Ensure PIL RGB
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    if model_choice == "ResNet-50 (TensorFlow)":
        # === ResNet preprocessing ===
        img = image.resize((224, 224), Image.BILINEAR)
        arr = tf.keras.utils.img_to_array(img)  # float32 [0..255], (224,224,3)

        if RESNET_INPUT_MODE == "manual_divide_255":
            arr = arr / 255.0
        elif RESNET_INPUT_MODE == "keras_app_resnet50":
            from tensorflow.keras.applications.resnet50 import preprocess_input
            arr = preprocess_input(arr)
        # else: "has_rescaling" -> the model itself rescales; do nothing

        arr = np.expand_dims(arr, axis=0)  # (1,224,224,3)

        raw = resnet_model.predict(arr, verbose=0)[0]  # (5,)
        raw = ensure_softmax(raw)
        preds = remap_probs_to_aptos(raw, RESNET_IDX_MAP)

    else:  # ViT (PyTorch)
        inputs = vit_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = vit_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # (5,)
        preds = remap_probs_to_aptos(probs, VIT_IDX_MAP)

    # Dict in APTOS order for Gradio Label
    return {label: float(preds[i]) for i, label in enumerate(APTOS_LABELS)}


# =========================================================
# 4) Gradio UI
# =========================================================
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(
            choices=["ViT (PyTorch)", "ResNet-50 (TensorFlow)"],
            value="ViT (PyTorch)",
            label="Choose a Model"
        ),
        gr.Image(type="pil", label="Upload a Retinal Fundus Image")
    ],
    outputs=gr.Label(num_top_classes=5, label="Predictions (APTOS order)"),
    title="Diabetic Retinopathy Classifier (ViT & ResNet)",
    description=(
        "Upload a fundus image and choose a model. "
        "Outputs are always shown in the APTOS clinical order: "
        "No_DR, Mild, Moderate, Severe, Proliferative_DR."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    # On Spaces, share=True is ignored; visibility is controlled in the Space settings.
    demo.launch()