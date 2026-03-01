import json
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "models/model.pt"
CLASS_PATH = "models/class_names.json"
IMG_SIZE = 224

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
uploaded = st.file_uploader("Upload an MRI image (jpg/png)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2, gap="large")

if uploaded:
    img = Image.open(uploaded)

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
        st.caption("Note: This demo is for educational purposes and not for clinical use.")
else:
    st.info("Upload an MRI image to see predictions.")
