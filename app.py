import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Skin Lesion Analyzer",
    layout="wide"
)

st.title("🩺 Skin Lesion Classification & Explainability")

# ---------------------------
# Device
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Class Names
# ---------------------------
CLASS_NAMES = ["BCC", "BKL", "MEL", "NV"]

# ---------------------------
# Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 4)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------------------
# Grad-CAM Implementation
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x):
        output = self.model(x)
        class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        confidence = torch.softmax(output, dim=1)[0, class_idx].item()

        return cam, class_idx, confidence

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload dermoscopic image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # -------- Display Uploaded Image (FIXED SIZE) --------
    st.subheader("Uploaded Image")
    st.image(image, width=300)

    # -------- Preprocess --------
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # -------- Grad-CAM --------
    gradcam = GradCAM(model, model.features[-1])
    cam, pred_class, confidence = gradcam.generate(input_tensor)

    # -------- Heatmap & Overlay --------
    img_np = np.array(image.resize((224, 224))) / 255.0
    cam_resized = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = 0.5 * img_np + 0.5 * heatmap

    # -------- Prediction Output --------
    st.subheader("Prediction")
    st.markdown(f"**Lesion Type:** `{CLASS_NAMES[pred_class]}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")

    # -------- Visualization Layout --------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Original")
        st.image(img_np, clamp=True)

    with col2:
        st.caption("Grad-CAM Heatmap")
        st.image(heatmap, clamp=True)

    with col3:
        st.caption("Overlay")
        st.image(overlay, clamp=True)
