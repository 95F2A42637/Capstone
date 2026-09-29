import os
import numpy as np
import torch
import streamlit as st

from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = r"models\resnet18_authenticity.pth"
IMAGE_SIZE = 224

st.set_page_config(
    page_title="Explainable AI Image Authenticity Verification",
    page_icon="🔍",
    layout="wide"
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    model = resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, class_names, device


# =========================================================
# IMAGE TRANSFORMATION
# =========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# IMAGE ANALYSIS + GRAD-CAM
# =========================================================

def analyze_image(image):

    model, class_names, device = load_model()

    original_image = image.convert("RGB")

    resized_image = original_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb_image = (
        np.array(resized_image).astype(np.float32) / 255.0
    )

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)

    # -----------------------------
    # Prediction
    # -----------------------------

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    predicted_label = class_names[predicted_class]

    # -----------------------------
    # Class probabilities
    # -----------------------------

    probability_dict = {}

    for index, class_name in enumerate(class_names):

        probability_dict[class_name] = (
            probabilities[0, index].item()
        )

    # -----------------------------
    # Grad-CAM
    # -----------------------------

    target_layers = [
        model.layer4[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return (
        predicted_label,
        confidence,
        probability_dict,
        visualization
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "🔍 Explainable AI Image Authenticity Verification"
)

st.write(
    "An AI-based system for detecting whether an image "
    "is authentic or AI-generated/manipulated, with "
    "visual explainability using Grad-CAM."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write(
        "**Model:** ResNet18"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )

    st.write(
        "**Input Size:** 224 × 224"
    )

    st.write(
        "**Classes:** Authentic / AI-Edited"
    )

    st.divider()

    st.header("📊 Dataset")

    st.metric(
        "Training Images",
        "16,000"
    )

    st.metric(
        "Validation Images",
        "2,000"
    )

    st.metric(
        "Test Images",
        "2,000"
    )

    st.divider()

    st.info(
        "The displayed evaluation values are based on "
        "the selected CIFAKE test set used during project development."
    )


# =========================================================
# NAVIGATION TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Image Verification",
    "📊 Model Performance",
    "ℹ️ About the System"
])


# =========================================================
# TAB 1 - IMAGE VERIFICATION
# =========================================================

with tab1:

    st.header(
        "🖼️ Image Authenticity Verification"
    )

    st.write(
        "Upload an image and analyze its predicted "
        "authenticity together with a Grad-CAM explanation."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        display_image = image.resize(
            (768, 768),
            Image.Resampling.LANCZOS
        )

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            display_image,
            use_container_width=True
        )

        if st.button(
            "🔎 Analyze Image",
            type="primary"
        ):

            with st.spinner(
                "Analyzing image using ResNet18..."
            ):

                try:

                    (
                        predicted_label,
                        confidence,
                        probability_dict,
                        visualization
                    ) = analyze_image(image)

                    st.divider()

                    st.subheader(
                        "🎯 Authenticity Verification Result"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Predicted Class",
                            predicted_label
                        )

                    with col2:

                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%"
                        )

                    with col3:

                        if predicted_label == "authentic":

                            result_text = "AUTHENTIC"

                        else:

                            result_text = "AI-EDITED"

                        st.metric(
                            "Decision",
                            result_text
                        )

                    # -------------------------
                    # Result message
                    # -------------------------

                    if predicted_label == "authentic":

                        st.success(
                            "The model predicts this image as AUTHENTIC."
                        )

                    else:

                        st.warning(
                            "The model predicts this image as AI-EDITED / MANIPULATED."
                        )

                    # -------------------------
                    # Class probabilities
                    # -------------------------

                    st.subheader(
                        "📈 Class Probability Distribution"
                    )

                    for class_name, probability in (
                        probability_dict.items()
                    ):

                        st.write(
                            f"**{class_name}** — "
                            f"{probability * 100:.2f}%"
                        )

                        st.progress(
                            float(probability)
                        )

                    st.divider()

                    # -------------------------
                    # Grad-CAM
                    # -------------------------

                    st.subheader(
                        "🔥 Explainability - Grad-CAM"
                    )

                    st.write(
                        "The heatmap highlights image regions "
                        "that contributed relatively more to "
                        "the model's prediction."
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.image(
                            image,
                            caption="Original Image",
                            use_container_width=True
                        )

                    with col2:

                        st.image(
                            visualization,
                            caption="Grad-CAM Explanation",
                           use_container_width=True
                        )

                    st.info(
                        "Grad-CAM provides an approximate visual "
                        "explanation of the model's decision. "
                        "It is not a pixel-level segmentation of edited regions."
                    )

                except Exception as e:

                    st.error(
                        f"Error during image analysis: {e}"
                    )


# =========================================================
# TAB 2 - MODEL PERFORMANCE
# =========================================================

with tab2:

    st.header(
        "📊 Model Performance"
    )

    st.write(
        "Evaluation results obtained during model development "
        "using the held-out test set."
    )

    # -----------------------------
    # Main metrics
    # -----------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Test Accuracy",
            "96.00%"
        )

    with col2:

        st.metric(
            "Precision",
            "96.00%"
        )

    with col3:

        st.metric(
            "Recall",
            "96.00%"
        )

    with col4:

        st.metric(
            "F1 Score",
            "96.00%"
        )

    st.divider()

    # -----------------------------
    # Training information
    # -----------------------------

    st.subheader(
        "🧠 Training Configuration"
    )

    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:

        st.write(
            "**Architecture:** ResNet18"
        )

        st.write(
            "**Image Size:** 224 × 224"
        )

    with config_col2:

        st.write(
            "**Training Epochs:** 3"
        )

        st.write(
            "**Batch Size:** 16"
        )

    with config_col3:

        st.write(
            "**Learning Rate:** 0.0001"
        )

        st.write(
            "**Optimizer:** AdamW"
        )

    st.divider()

    # -----------------------------
    # Dataset distribution
    # -----------------------------

    st.subheader(
        "📁 Dataset Distribution"
    )

    dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

    with dataset_col1:

        st.metric(
            "Training",
            "16,000 images"
        )

    with dataset_col2:

        st.metric(
            "Validation",
            "2,000 images"
        )

    with dataset_col3:

        st.metric(
            "Testing",
            "2,000 images"
        )

    st.divider()

    # -----------------------------
    # Evaluation images
    # -----------------------------

    st.subheader(
        "📈 Training and Evaluation Graphs"
    )

    accuracy_path = r"outputs\accuracy_curve.png"
    loss_path = r"outputs\loss_curve.png"
    confusion_path = r"outputs\confusion_matrix.png"

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:

        if os.path.exists(accuracy_path):

            st.image(
                accuracy_path,
                caption="Training and Validation Accuracy"
            )

        else:

            st.warning(
                "Accuracy curve not found."
            )

    with graph_col2:

        if os.path.exists(loss_path):

            st.image(
                loss_path,
                caption="Training and Validation Loss"
            )

        else:

            st.warning(
                "Loss curve not found."
            )

    st.divider()

    st.subheader(
        "🔲 Confusion Matrix"
    )

    if os.path.exists(confusion_path):

        st.image(
            confusion_path,
            caption="Test Set Confusion Matrix",
            width=650
        )

    else:

        st.warning(
            "Confusion matrix not found."
        )

    st.divider()

    st.subheader(
        "📄 Classification Report"
    )

    report_path = r"outputs\test_report.txt"

    if os.path.exists(report_path):

        with open(
            report_path,
            "r",
            encoding="utf-8"
        ) as file:

            report_text = file.read()

        st.code(
            report_text,
            language="text"
        )

    else:

        st.warning(
            "Classification report not found."
        )


# =========================================================
# TAB 3 - ABOUT THE SYSTEM
# =========================================================

with tab3:

    st.header(
        "ℹ️ About the System"
    )

    st.subheader(
        "🎯 Objective"
    )

    st.write(
        "The objective of this project is to develop an "
        "Explainable AI system that predicts whether an "
        "input image is authentic or AI-generated/manipulated "
        "and provides a visual explanation for the prediction."
    )

    st.subheader(
        "⚙️ System Workflow"
    )

    st.write(
        "1. User uploads an image"
    )

    st.write(
        "2. Image is resized and normalized"
    )

    st.write(
        "3. ResNet18 extracts visual features"
    )

    st.write(
        "4. Classification layer predicts authenticity"
    )

    st.write(
        "5. Confidence score is calculated"
    )

    st.write(
        "6. Grad-CAM generates an approximate visual explanation"
    )

    st.subheader(
        "🧠 Technologies Used"
    )

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:

        st.write("• Python")
        st.write("• PyTorch")
        st.write("• Torchvision")
        st.write("• ResNet18")

    with tech_col2:

        st.write("• Streamlit")
        st.write("• Grad-CAM")
        st.write("• NumPy")
        st.write("• Pillow")

    st.subheader(
        "⚠️ Limitations"
    )

    st.write(
        "• Model performance depends on the training dataset."
    )

    st.write(
        "• Predictions may contain false positives or false negatives."
    )

    st.write(
        "• Grad-CAM provides approximate decision explanations "
        "rather than exact edited-pixel segmentation."
    )

    st.write(
        "• Real-world images may differ from the dataset distribution."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Explainable AI Image Authenticity Verification | "
    "ResNet18 + Grad-CAM | Final Year Project"
)