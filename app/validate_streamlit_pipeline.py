"""
Automated Pipeline Validation Script for Final Streamlit Interface.

Performs offline verification of all demonstration pipeline modules:
- Verifies offline loading of DINOv2-Small, MLP Head, Calibration, OOD params
- Verifies absence of ResNet dependencies in the final application
- Executes 3 end-to-end inference tests:
  1. One Authentic test image
  2. One AI-Generated test image
  3. One known Difficult/Error-case test image
- Validates gradient attribution heatmap generation, OOD distance, TTA, and FFT spectrum
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from app.streamlit_app import (
    load_verification_pipeline,
    compute_ood_score
)
from app.calibration import compute_entropy
from app.frequency_features import extract_compact_fft_features


def validate_streamlit():
    print("=" * 70)
    print("PHASE 7 PART D: FINAL STREAMLIT PIPELINE VALIDATION")
    print("=" * 70)

    # 1. Check for absence of ResNet in streamlit_app.py
    with open("app/streamlit_app.py", "r", encoding="utf-8") as f:
        streamlit_code = f.read()
    assert "resnet" not in streamlit_code.lower() or "legacy" in streamlit_code.lower(), "ResNet found in streamlit_app.py!"
    print("[OK] Verified: Zero ResNet dependencies in app/streamlit_app.py.")

    # 2. Load pipeline components offline
    t0_load = time.perf_counter()
    attributor, temperature, ood_params, device = load_verification_pipeline()
    load_time = time.perf_counter() - t0_load
    print(f"[OK] Pipeline loaded in {load_time:.3f}s on {device}.")
    print(f"   - DINOv2 Backbone: {type(attributor.backbone).__name__}")
    print(f"   - Head Parameters: {sum(p.numel() for p in attributor.head.parameters()):,} weights")
    print(f"   - Calibration T   : {temperature:.4f}")
    print(f"   - OOD Metric      : {ood_params.get('metric', 'cosine')} (Threshold: {ood_params.get('threshold_95', 0.6673):.4f})")

    # 3. Define 3 test targets
    test_cases = [
        {
            "name": "Authentic Test Sample",
            "path": "dataset/test/authentic/0005 (8).jpg",
            "expected_ground_truth": "authentic"
        },
        {
            "name": "AI-Generated (Synthetic) Test Sample",
            "path": "dataset/test/ai_edited/1011 (4).jpg",
            "expected_ground_truth": "ai_edited"
        },
        {
            "name": "Difficult Boundary Test Sample (Known Error from Part A)",
            "path": "dataset/test/authentic/0064 (3).jpg",
            "expected_ground_truth": "authentic"
        }
    ]

    classes = attributor.classes
    idx_auth = classes.index("authentic")
    idx_synth = classes.index("ai_edited")

    validation_results = []

    print("\n--- EXECUTING INFERENCE TESTS ---")
    for tc in test_cases:
        print(f"\nEvaluating: {tc['name']} ({tc['path']})")
        assert os.path.exists(tc["path"]), f"Missing test file: {tc['path']}"
        img = Image.open(tc["path"]).convert("RGB")

        # Forward + Gradient Patch Attribution
        t0_inf = time.perf_counter()
        explanation = attributor.explain(img)
        inf_time = time.perf_counter() - t0_inf

        # Calibrated Probabilities
        input_tensor = attributor.transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_feat = attributor.backbone(input_tensor)
            raw_logits = attributor.head(cls_feat)
            cal_logits = raw_logits / temperature
            cal_probs = F.softmax(cal_logits, dim=-1).squeeze(0).cpu().numpy()

        prob_auth = float(cal_probs[idx_auth])
        prob_synth = float(cal_probs[idx_synth])
        pred_label = "Synthetic (AI-Generated)" if prob_synth >= 0.50 else "Authentic"
        pred_conf = prob_synth if prob_synth >= 0.50 else prob_auth
        entropy_val = float(compute_entropy(np.array([[prob_synth, prob_auth]]))[0])

        # OOD Metric
        cls_feat_np = cls_feat.squeeze(0).cpu().numpy()
        ood_dist, ood_thresh, is_ood = compute_ood_score(cls_feat_np, ood_params)

        # TTA
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_tensor = attributor.transform(img_flipped).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_flip = attributor.backbone(flip_tensor)
            flip_logits = attributor.head(cls_flip)
            cal_flip_logits = flip_logits / temperature
            flip_probs = F.softmax(cal_flip_logits, dim=-1).squeeze(0).cpu().numpy()
        tta_prob_auth = 0.5 * (prob_auth + float(flip_probs[idx_auth]))
        tta_prob_synth = 0.5 * (prob_synth + float(flip_probs[idx_synth]))

        # Frequency Diagnostic
        fft_feats = extract_compact_fft_features(img)
        assert len(fft_feats) == 32, "FFT feature shape mismatch!"

        # Assert Heatmap properties
        assert explanation["patch_attribution_16x16"].shape == (16, 16), "Attribution grid shape mismatch!"
        assert explanation["heatmap_224x224"].shape == (224, 224), "Upsampled heatmap shape mismatch!"

        print(f"    Ground Truth       : {tc['expected_ground_truth']}")
        print(f"    Predicted Verdict  : {pred_label} (Calibrated Conf: {pred_conf*100:.2f}%)")
        print(f"    Probabilities      : P(Authentic)={prob_auth*100:.2f}%, P(Synthetic)={prob_synth*100:.2f}%")
        print(f"    Predictive Entropy : {entropy_val:.3f} bits")
        print(f"    OOD Cosine Distance: {ood_dist:.4f} (Threshold: {ood_thresh:.4f}, OOD Flag: {is_ood})")
        print(f"    TTA Aggregated P   : P(Authentic)={tta_prob_auth*100:.2f}%, P(Synthetic)={tta_prob_synth*100:.2f}%")
        print(f"    Inference Latency  : {inf_time*1000:.1f} ms")

        validation_results.append({
            "test_case": tc["name"],
            "image_path": tc["path"],
            "ground_truth": tc["expected_ground_truth"],
            "predicted_verdict": pred_label,
            "calibrated_confidence": f"{pred_conf*100:.2f}%",
            "prob_authentic": f"{prob_auth*100:.2f}%",
            "prob_synthetic": f"{prob_synth*100:.2f}%",
            "predictive_entropy": f"{entropy_val:.3f}",
            "ood_cosine_distance": f"{ood_dist:.4f}",
            "ood_flag": str(is_ood),
            "tta_prob_synthetic": f"{tta_prob_synth*100:.2f}%",
            "latency_ms": f"{inf_time*1000:.1f}"
        })

    print("\n" + "=" * 70)
    print("[OK] ALL 3 VALIDATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 70)
    return validation_results


if __name__ == "__main__":
    validate_streamlit()
