import streamlit as st
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image

# =========================================================
# Streamlit Page Configuration
# =========================================================
st.set_page_config(page_title="Breast Image Segmentation", layout="wide")
st.title("🧠 Breast Image Segmentation Demo")
st.markdown(
    "Upload a breast image, select a model (VGG16 or ResNet50), "
    "and visualize the original, label, and predicted segmentation."
)

# =========================================================
# Model Loader
# =========================================================
@st.cache_resource
def load_model(model_name):
    """Load the selected UNet model with pretrained weights."""
    if model_name == "ResNet50":
        weights_path = "V-Unet/pytorch/after_aug/decoder_only/best_dice_res50_unet_30epoch_aa_de.pth"
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,  # don’t reload ImageNet weights at inference
            in_channels=3,
            classes=1,
        )
    elif model_name == "VGG16":
        weights_path = "V-Unet/pytorch/after_aug/decoder_only/best_dice_vgg16_unet_30epoch_aa_de.pth"
        model = smp.Unet(
            encoder_name="vgg16",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load the saved state_dict (weights)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =========================================================
# Image Preprocessing and Prediction
# =========================================================
def preprocess_image(img: Image.Image):
    """Resize and normalize image for model input."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor


def predict_mask(model, img):
    """Run model inference and return predicted mask."""
    x = preprocess_image(img)
    with torch.no_grad():
        pred = model(x)
        pred_mask = torch.sigmoid(pred).squeeze().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    return Image.fromarray(pred_mask)


# =========================================================
# Streamlit UI
# =========================================================
uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
model_choice = st.selectbox("Select model", ["VGG16", "ResNet50"])

# Optional: Upload label image for comparison
label_file = st.file_uploader("📥 (Optional) Upload label/mask", type=["png", "jpg", "jpeg"])

if uploaded_file:
    original_img = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col1.image(original_img, caption="Original Image", use_container_width=True)

    if label_file:
        label_img = Image.open(label_file).convert("L")
        col2.image(label_img, caption="Ground Truth Label", use_container_width=True)
    else:
        col2.markdown("🩻 *No label uploaded*")

    if st.button("Run Prediction 🚀"):
        with st.spinner(f"Loading {model_choice} model..."):
            model = load_model(model_choice)

        with st.spinner("Running prediction..."):
            pred_img = predict_mask(model, original_img)

        col3.image(pred_img, caption="Predicted Mask", use_container_width=True)
else:
    st.info("👆 Upload an image file to begin.")