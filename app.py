import streamlit as st
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import os

# =========================================================
# Streamlit Page Configuration
# =========================================================
st.set_page_config(page_title="Breast Image Segmentation", layout="wide")
st.title("🧠 Breast Image Segmentation Demo")
st.markdown(
    "Upload a breast image, select a model variant, "
    "and visualize the original, label, and predicted segmentation."
)

# =========================================================
# Model Path & Loader
# =========================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_OPTIONS = {
    "ResNet50 UNet — after aug (decoder only)": "res50_unet_aa_de.pth",
    "ResNet50 UNet — before aug (include encoder)": "res50_unet_ba_en.pth",
    "ResNet50 UNet++ — after aug (include encoder)": "res50_unet++_aa_en.pth",
    "ResNet50 UNet++ — before aug (decoder only)": "res50_unet++_ba_de.pth",
    "VGG16 UNet — before aug (decoder only)": "vgg16_unet_ba_de.pth",
    "VGG16 UNet++ — after aug (decoder only)": "vgg16_unet++_aa_de.pth",
    "Xception UNet — after aug (include encoder)": "Xception_unet_aa_en.pth",
    "Xception UNet — before aug (include encoder)": "Xception_unet_ba_en.pth",
}

@st.cache_resource
def load_model(model_key):
    """Load the selected model with pretrained weights."""
    filename = MODEL_OPTIONS[model_key]
    weights_path = os.path.join(MODEL_DIR, filename)

    if "res50_unet++" in filename:
        model_class = smp.UnetPlusPlus
        encoder = "resnet50"
    elif "res50_unet" in filename:
        model_class = smp.Unet
        encoder = "resnet50"
    elif "vgg16_unet++" in filename:
        model_class = smp.UnetPlusPlus
        encoder = "vgg16"
    elif "vgg16_unet" in filename:
        model_class = smp.Unet
        encoder = "vgg16"
    elif "Xception_unet" in filename:
        model_class = smp.Unet
        encoder = "xception"
    else:
        raise ValueError("Unknown model file name pattern.")

    model = model_class(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# =========================================================
# Image Preprocessing and Prediction
# =========================================================
def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def predict_mask(model, img):
    # store original size
    orig_size = img.size  # (width, height)

    x = preprocess_image(img)

    with torch.no_grad():
        pred = model(x)
        pred_mask = torch.sigmoid(pred).squeeze().numpy()

    # binarize mask
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # resize back to original size
    pred_mask_img = Image.fromarray(pred_mask).resize(orig_size, resample=Image.NEAREST)
    return pred_mask_img


# =========================================================
# Streamlit UI
# =========================================================
# Sidebar for dataset folder input, this si for auto label lookup
st.sidebar.header("🗂️ Dataset Folder (for auto label lookup)")
dataset_dir = st.sidebar.text_input("Enter dataset folder path (e.g. D:/breast_dataset):")

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
model_choice = st.selectbox("Select model variant", list(MODEL_OPTIONS.keys()))

if uploaded_file:
    # save uploaded image temporarily
    temp_dir = "uploaded_temp"
    os.makedirs(temp_dir, exist_ok=True)
    img_path = os.path.join(temp_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # show original image
    original_img = Image.open(img_path).convert("RGB")

    # get mask automatically if exists
    mask_img = None
    if dataset_dir and os.path.isdir(dataset_dir):
        base_name, _ = os.path.splitext(uploaded_file.name)
        candidate_mask = f"{base_name}_mask.png"
        mask_path = os.path.join(dataset_dir, candidate_mask)
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")

    # show plot
    col1, col2, col3 = st.columns(3)
    col1.image(original_img, caption="Original Image", use_container_width=True)

    if mask_img:
        col2.image(mask_img, caption="Ground Truth Label (Auto-loaded)", use_container_width=True)
    else:
        col2.markdown("🩻 *No matching label found in dataset folder.*")

    # prediction
    if st.button("Run Prediction 🚀"):
        with st.spinner(f"Loading {model_choice}..."):
            model = load_model(model_choice)

        with st.spinner("Running prediction..."):
            pred_img = predict_mask(model, original_img)

        col3.image(pred_img, caption="Predicted Mask", use_container_width=True)
else:
    st.info("👆 Upload an image file to begin.")
