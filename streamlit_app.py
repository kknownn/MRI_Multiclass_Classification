import json
import numpy as np
import streamlit as st
from PIL import Image
import os
import random
from glob import glob

import torch
import torch.nn as nn
from torchvision import models, transforms

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "models/model.pt"
CLASS_PATH = "models/class_names.json"
IMG_SIZE = 224
# Optional: local sample dataset path (for random demo images)
# Use Testing folder for random demo samples
SAMPLE_DATASET_DIR = "dataset/Testing"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
st.title("Brain Tumor MRI Classifier (ResNet50, PyTorch)")

# ----------------------------
# Device (Apple Silicon MPS)
# ----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
st.caption(f"Device: `{device}`")

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_class_names(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model(model_path: str, num_classes: int):
    # Must match the training architecture exactly
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    model.to(device)
    return model

class_names = load_class_names(CLASS_PATH)
num_classes = len(class_names)
model = load_model(MODEL_PATH, num_classes)

# ----------------------------
# Preprocessing
# ----------------------------
eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def predict_pil(img: Image.Image):
    img = img.convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top = int(np.argmax(probs))
    return probs, top

# ----------------------------
# UI
# ----------------------------
st.subheader("Choose Image Source")
mode = st.radio(
    "Select input method:",
    ["Upload Image", "Random Sample from Dataset"]
)

uploaded = None
sample_image = None

if mode == "Upload Image":
    uploaded = st.file_uploader("Upload an MRI image (jpg/png)", type=["jpg", "jpeg", "png"])

elif mode == "Random Sample from Dataset":
    if os.path.isdir(SAMPLE_DATASET_DIR):
        image_paths = glob(os.path.join(SAMPLE_DATASET_DIR, "**", "*.jpg"), recursive=True)
        image_paths += glob(os.path.join(SAMPLE_DATASET_DIR, "**", "*.png"), recursive=True)

        if image_paths:
            col_a, col_b = st.columns(2)

            if col_a.button("Load Random Sample"):
                sample_path = random.choice(image_paths)
                st.session_state["sample_path"] = sample_path

            if col_b.button("Random Again"):
                sample_path = random.choice(image_paths)
                st.session_state["sample_path"] = sample_path

            if "sample_path" in st.session_state:
                sample_path = st.session_state["sample_path"]
                sample_image = Image.open(sample_path)

                # Extract metadata
                file_name = os.path.basename(sample_path)
                file_ext = os.path.splitext(sample_path)[1]
                true_label = os.path.basename(os.path.dirname(sample_path))

                st.caption(f"File: {file_name}")
                st.caption(f"Type: {file_ext}")
                st.caption(f"True Label (folder): {true_label}")
        else:
            st.warning("No images found in SAMPLE_DATASET_DIR.")
    else:
        st.warning("SAMPLE_DATASET_DIR does not exist. Update the path in the config.")

image_to_use = None

if uploaded:
    image_to_use = Image.open(uploaded)

elif sample_image is not None:
    image_to_use = sample_image

if image_to_use is not None:
    img = image_to_use

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)

    with col2:
        auto = st.toggle("Auto-predict", value=True)
        run = st.button("Predict", type="primary", disabled=auto)

    if auto or run:
        probs, top = predict_pil(img)

        pred_label = class_names[top]
        conf = float(probs[top])

        st.subheader(f"Prediction: **{pred_label}**")
        st.write(f"Confidence: **{conf:.2%}**")

        top2 = probs.argsort()[-2:][::-1]
        st.write("Top-2:")
        st.write(f"- {class_names[int(top2[0])]} — {float(probs[int(top2[0])]):.2%}")
        st.write(f"- {class_names[int(top2[1])]} — {float(probs[int(top2[1])]):.2%}")

        st.bar_chart({class_names[i]: float(probs[i]) for i in range(num_classes)})

        st.divider()
        st.caption("Note: This demo is for educational purposes, submission for 888351 | Modern Computer Vision And Applications For Entrepreneur. And not for clinical use.")
else:
    st.info("Upload an MRI image to see predictions.")
