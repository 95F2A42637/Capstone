"""
Streamlit Demonstration Application for Vision Foundation Model Authenticity Verification.

Project Title:
Explainable AI-Based Image Authenticity Verification and AI Edit Detection using Vision Foundation Models

Pipeline:
Input Image -> DINOv2-Small (ViT-S/14) -> 384-D Representation -> MLP Head
-> Calibrated Confidence -> Predictive Entropy -> Gradient Patch Attribution
-> Representation OOD Indicator -> Optional TTA -> Frequency Diagnostics
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import streamlit as st

from app.models_vfm import (
    load_dinov2_small,
    get_dinov2_transforms,
    DINOv2ClassificationHead,
    DINOv2AuthenticityClassifier,
    EMBEDDING_DIM,
    DEFAULT_IMAGE_SIZE
)
from app.explainability import DINOv2PatchAttributor
from app.calibration import compute_entropy
from app.frequency_features import extract_compact_fft_features


# Configure page layout
st.set_page_config(
    page_title="Image Authenticity Verification (DINOv2)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_verification_pipeline():
    """Caches pipeline resources for fast subsequent inference."""
    device = torch.device("cpu")

    # 1. Explainability & Classification Attributor
    attributor = DINOv2PatchAttributor(
        checkpoint_path="models/dinov2_authenticity_head.pth",
        device=device
    )

    # 2. Calibration Parameters
    calib_path = "models/calibration_params.json"
    temp = 1.0
    if os.path.exists(calib_path):
        with open(calib_path, "r", encoding="utf-8") as f:
            cdata = json.load(f)
            temp = float(cdata.get("temperature", 1.0))

    # 3. OOD Parameters
    ood_path = "models/ood_params.json"
    ood_params = None
    if os.path.exists(ood_path):
        with open(ood_path, "r", encoding="utf-8") as f:
            ood_params = json.load(f)

    return attributor, temp, ood_params, device


def compute_ood_score(features_np: np.ndarray, ood_params: dict):
    """Computes minimum cosine distance to training class centroids."""
    if ood_params is None:
        return 0.0, 0.6673, False

    mu_auth = np.array(ood_params["centroid_authentic"], dtype=np.float32)
    mu_synth = np.array(ood_params["centroid_synthetic"], dtype=np.float32)

    feat_norm = np.linalg.norm(features_np) + 1e-12
    cos_auth = np.dot(features_np, mu_auth) / (feat_norm * (np.linalg.norm(mu_auth) + 1e-12))
    cos_synth = np.dot(features_np, mu_synth) / (feat_norm * (np.linalg.norm(mu_synth) + 1e-12))

    dist_auth = 1.0 - float(cos_auth)
    dist_synth = 1.0 - float(cos_synth)
    min_dist = min(dist_auth, dist_synth)

    threshold = float(ood_params.get("threshold_95", 0.6673))
    is_ood = (min_dist > threshold)
    return min_dist, threshold, is_ood


def main():
    # 1. Header
    st.title("🛡️ Image Authenticity Verification & AI Edit Detection")
    st.subheader("Research Demonstration Interface | Vision Foundation Model (DINOv2-Small)")
    st.markdown("---")

    attributor, temperature, ood_params, device = load_verification_pipeline()

    # 2. Sidebar Controls
    st.sidebar.header("📁 Image Selection")
    source_choice = st.sidebar.radio("Input Source", ["Upload Custom Image", "Benchmark Test Preset"])

    img_input = None

    if source_choice == "Upload Custom Image":
        uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img_input = Image.open(uploaded_file).convert("RGB")
    else:
        preset_options = {
            "Authentic Image (Test Set)": os.path.join("dataset", "test", "authentic", "0005 (8).jpg"),
            "AI-Generated Image (Test Set)": os.path.join("dataset", "test", "ai_edited", "1011 (4).jpg"),
            "Difficult/Borderline Case (Authentic misclassified)": os.path.join("dataset", "test", "authentic", "0064 (3).jpg"),
            "Difficult/Borderline Case (Synthetic misclassified)": os.path.join("dataset", "test", "ai_edited", "1040.jpg")
        }
        selected_preset = st.sidebar.selectbox("Select Test Image", list(preset_options.keys()))
        preset_path = preset_options[selected_preset]
        if os.path.exists(preset_path):
            img_input = Image.open(preset_path).convert("RGB")
            st.sidebar.caption(f"Path: `{preset_path}`")
        else:
            st.sidebar.error("Preset file not found.")

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Evaluation Options")
    enable_tta = st.sidebar.checkbox("Enable Test-Time Augmentation (TTA)", value=False,
                                    help="Averages predictions of original and horizontally flipped views. Approximately doubles CPU latency.")
    show_frequency = st.sidebar.checkbox("Show Diagnostic Frequency Spectrum (FFT)", value=True,
                                         help="Extracts 2D FFT magnitude spectrum and radial energy distribution as an orthogonal diagnostic.")

    if img_input is None:
        st.info("👆 Please upload an image or select a benchmark preset from the sidebar to begin analysis.")
        return

    # Display Input Image
    col_input, col_meta = st.columns([1, 2])
    with col_input:
        st.image(img_input, caption="Input Query Image", use_container_width=True)
    with col_meta:
        st.markdown(f"**Image Dimensions:** {img_input.width} × {img_input.height} px")
        st.markdown(f"**Standardized Backbone Input:** 224 × 224 px")
        st.markdown(f"**Backbone:** `facebook/dinov2-small` (Frozen ViT-S/14)")
        st.markdown(f"**Calibration:** Temperature Scaling ($T={temperature:.4f}$)")

    st.markdown("---")

    # Run Analysis
    with st.spinner("Executing DINOv2 foundation model inference and gradient attribution..."):
        t0 = time.perf_counter()

        # Step A: Gradient Patch Attribution & Prediction
        explanation = attributor.explain(img_input)
        total_time = (time.perf_counter() - t0) * 1000

        # Step B: Calibrated Probabilities
        raw_logits_t = torch.tensor([
            explanation["probabilities"]["ai_edited"],
            explanation["probabilities"]["authentic"]
        ], dtype=torch.float32)
        # Note: explanation["probabilities"] are raw softmax probabilities from attributor
        # Extract raw logits from the head
        input_tensor = attributor.transform(img_input).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_feat = attributor.backbone(input_tensor)  # [1, 384]
            raw_logits = attributor.head(cls_feat)  # [1, 2]
            cal_logits = raw_logits / temperature
            cal_probs = F.softmax(cal_logits, dim=-1).squeeze(0).cpu().numpy()

        classes = attributor.classes
        idx_auth = classes.index("authentic")
        idx_synth = classes.index("ai_edited")

        prob_auth = float(cal_probs[idx_auth])
        prob_synth = float(cal_probs[idx_synth])

        # Binary Decision
        if prob_synth >= 0.50:
            pred_label = "AI-Generated / Synthetic"
            pred_conf = prob_synth
            badge_color = "red"
        else:
            pred_label = "Authentic"
            pred_conf = prob_auth
            badge_color = "green"

        entropy_val = float(compute_entropy(np.array([[prob_synth, prob_auth]]))[0])

        # Step C: OOD Cosine Distance
        cls_feat_np = cls_feat.squeeze(0).cpu().numpy()
        ood_dist, ood_thresh, is_ood = compute_ood_score(cls_feat_np, ood_params)

        # Step D: Optional TTA
        tta_executed = False
        tta_prob_auth = None
        tta_prob_synth = None
        if enable_tta:
            t0_tta = time.perf_counter()
            img_flipped = img_input.transpose(Image.FLIP_LEFT_RIGHT)
            flip_tensor = attributor.transform(img_flipped).unsqueeze(0).to(device)
            with torch.no_grad():
                cls_flip = attributor.backbone(flip_tensor)
                flip_logits = attributor.head(cls_flip)
                cal_flip_logits = flip_logits / temperature
                flip_probs = F.softmax(cal_flip_logits, dim=-1).squeeze(0).cpu().numpy()
            tta_time_ms = (time.perf_counter() - t0_tta) * 1000
            tta_prob_auth = 0.5 * (prob_auth + float(flip_probs[idx_auth]))
            tta_prob_synth = 0.5 * (prob_synth + float(flip_probs[idx_synth]))
            tta_executed = True

    # 3. Authenticity Result Section
    st.subheader("📊 Authenticity Verification Verdict")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        if pred_label == "Authentic":
            st.success(f"**Verdict:** {pred_label}")
        else:
            st.error(f"**Verdict:** {pred_label}")

    with m2:
        st.metric(
            label="Calibrated Confidence",
            value=f"{pred_conf * 100:.2f}%",
            help="Post-hoc temperature-scaled probability (T=0.9464) on the held-out validation set."
        )

    with m3:
        st.metric(
            label="Predictive Shannon Entropy",
            value=f"{entropy_val:.3f} bits",
            help="Uncertainty metric: Low entropy (<0.3 bits) indicates high certainty; elevated entropy (>0.7 bits) indicates ambiguity."
        )

    with m4:
        st.metric(
            label="Inference Latency",
            value=f"{total_time:.1f} ms",
            help="Complete forward pass and gradient backpropagation latency on CPU."
        )

    # Probability bars
    c_bar1, c_bar2 = st.columns(2)
    with c_bar1:
        st.write(f"**Authentic Probability:** `{prob_auth * 100:.2f}%`")
        st.progress(float(prob_auth))
    with c_bar2:
        st.write(f"**Synthetic Probability:** `{prob_synth * 100:.2f}%`")
        st.progress(float(prob_synth))

    # 4. Explainability Section
    st.markdown("---")
    st.subheader("🔍 Spatial Model Attribution (Explainability)")
    st.markdown(
        "*Gradient-based patch attribution showing image regions that contributed more strongly to the model decision. "
        "This is an approximate model attribution and is **not** a ground-truth tampering mask.*"
    )

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.image(img_input.resize((224, 224)), caption="Standardized Input (224×224)", use_container_width=True)
    with exp_col2:
        st.image(explanation["colored_heatmap"], caption="Patch Relevance Heatmap (16×16 Grid, Bicubic)", use_container_width=True)
    with exp_col3:
        st.image(explanation["overlay_image"], caption="Attribution Overlay on Input", use_container_width=True)

    # 5. Out-of-Distribution (OOD) Indicator
    st.markdown("---")
    st.subheader("🛰️ Out-of-Distribution (OOD) Indicator")
    st.markdown(
        "*Evaluates whether the 384-dimensional DINOv2 feature representation departs from the training distribution "
        "using class-centroid minimum cosine distance.*"
    )

    ood_col1, ood_col2 = st.columns([1, 2])
    with ood_col1:
        st.metric(label="Centroid Cosine Distance", value=f"{ood_dist:.4f}")
        st.caption(f"Calibrated 95th Percentile Threshold: **{ood_thresh:.4f}**")

    with ood_col2:
        if is_ood:
            st.warning(
                f"⚠️ **Atypical Representation (OOD Flag):** The query feature distance ({ood_dist:.4f}) exceeds the "
                f"95% validation threshold ({ood_thresh:.4f}).\n\n"
                "*Notice: An OOD warning indicates that the image lies in an atypical region of the feature space relative to "
                "the training dataset. It does NOT automatically prove the image is synthetic or tampered.*"
            )
        else:
            st.success(
                f"✅ **In-Distribution Representation:** The query feature distance ({ood_dist:.4f}) is within the 95th percentile "
                f"in-distribution threshold ({ood_thresh:.4f})."
            )

    # 6. Optional TTA Comparison Section
    if tta_executed:
        st.markdown("---")
        st.subheader("🔄 Test-Time Augmentation (TTA) Comparison")
        st.markdown(
            "*Two-view probability aggregation ($0.50 \\times \\text{Original} + 0.50 \\times \\text{Horizontal Flip}$). "
            "Phase 6 benchmarks demonstrate a $+0.40\\%$ accuracy shift at a $2.01\\times$ CPU latency overhead.*"
        )
        tta_c1, tta_c2, tta_c3 = st.columns(3)
        with tta_c1:
            st.markdown("**Standard Single-Crop:**")
            st.write(f"- P(Authentic): `{prob_auth * 100:.2f}%`")
            st.write(f"- P(Synthetic): `{prob_synth * 100:.2f}%`")
        with tta_c2:
            st.markdown("**TTA (Averaged Views):**")
            st.write(f"- P(Authentic): `{tta_prob_auth * 100:.2f}%`")
            st.write(f"- P(Synthetic): `{tta_prob_synth * 100:.2f}%`")
        with tta_c3:
            delta = (tta_prob_synth - prob_synth) * 100
            st.metric(
                label="Synthetic Probability Delta",
                value=f"{delta:+.2f}%",
                help="Difference between TTA probability and single-crop probability."
            )

    # 7. Diagnostic Frequency Analysis
    if show_frequency:
        st.markdown("---")
        st.subheader("⚡ Frequency-Domain Diagnostic Analysis")
        st.caption("Diagnostic frequency-domain visualization — not used as the primary final classifier.")

        fft_gray = img_input.convert("L").resize((224, 224))
        fft_arr = np.asarray(fft_gray, dtype=np.float64) / 255.0
        fft_2d = np.fft.fftshift(np.fft.fft2(fft_arr))
        log_mag = np.log1p(np.abs(fft_2d))
        norm_spec = ((log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-12) * 255).astype(np.uint8)

        # 32-D FFT compact features
        fft_feats = extract_compact_fft_features(img_input)
        radial_8 = fft_feats[:8]

        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            st.image(norm_spec, caption="2D FFT Log-Magnitude Spectrum (DC Centered)", use_container_width=True)
        with f_col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(1, 9), radial_8 * 100, color="teal", alpha=0.85)
            ax.set_xlabel("Concentric Radial Frequency Band (1=Low, 8=Nyquist High)", fontsize=9)
            ax.set_ylabel("Power Proportion (%)", fontsize=9)
            ax.set_title("Radial Spectral Energy Partition (8 Bands)", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

    # 8. Research Disclaimer
    st.markdown("---")
    st.warning(
        "⚠️ **Research Disclaimer:** This system provides an AI-based authenticity assessment, not definitive forensic proof. "
        "The explainability visualization represents approximate model attribution rather than ground-truth manipulation localization. "
        "OOD detection has not been validated on an external real-world OOD benchmark in this project."
    )

    # 9. Technical Details (Expandable)
    with st.expander("🛠️ System Technical Details & Specifications"):
        st.markdown("""
        - **Primary Foundation Model:** DINOv2-Small (`facebook/dinov2-small` / `ViT-S/14`)
        - **Parameters:** 21M backbone parameters (frozen during training and inference)
        - **Input Resolution:** $224 \\times 224$ pixels, standardized with ImageNet normalization
        - **Representation:** 384-dimensional CLS token representation + 256 spatial patch tokens ($16 \\times 16$ grid)
        - **Classification Head:** LayerNorm $\\rightarrow$ Linear(384, 256) $\\rightarrow$ ReLU $\\rightarrow$ Dropout(0.1) $\\rightarrow$ Linear(256, 2)
        - **Confidence Calibration:** Validation-set post-hoc temperature scaling ($T=0.9464$), reducing ECE from 1.00% to 0.73%
        - **Explainability Engine:** Gradient backpropagation through Block 12 w.r.t. patch tokens with bicubic upsampling
        - **OOD Indicator:** Minimum class-centroid cosine distance in 384-D representation space with 95th percentile validation thresholding
        - **Execution Hardware:** Pure CPU inference pipeline, optimized for local academic deployment
        """)

    # 10. Model Limitations
    with st.expander("⚠️ Known Model Limitations"):
        st.markdown("""
        1. **Dataset Scope:** The classifier was trained and evaluated on CIFAKE (Stable Diffusion v1.4 vs CIFAR-10 natural imagery). Generalization to commercial generators (Midjourney v6, DALL-E 3, Flux) cannot be guaranteed.
        2. **Absence of Ground-Truth Masks:** CIFAKE contains whole-image synthetic samples; gradient heatmaps show where the model focused, not human-annotated tampered pixels.
        3. **High-Frequency Degradation:** As demonstrated in Phase 6 robustness testing, extreme blur ($r=3.0$, accuracy drop to 68.4%) and lossy downscaling ($0.25\\times$, accuracy drop to 70.0%) significantly impair detection.
        4. **OOD Benchmark Unavailability:** Lack of a third-party out-of-distribution benchmark locally restricts OOD verification to proxy manifold testing.
        5. **TTA Overhead:** Test-time augmentation approximately doubles CPU latency ($2.01\\times$) for marginal accuracy gains ($+0.40\\%$).
        """)


if __name__ == "__main__":
    main()
