"""
Deterministic Multi-Perturbation Robustness Evaluation Suite.

Evaluates how the trained DINOv2 authenticity classifier behaves when held-out
test images are subjected to 6 controlled image perturbations across 3 severity levels:
1. JPEG Compression (Q=75, Q=50, Q=25)
2. Downscale-Upscale Resizing (Factor=0.75, Factor=0.50, Factor=0.25)
3. Gaussian Blur (Radius=1.0, Radius=2.0, Radius=3.0)
4. Brightness Modification (Factor=0.8, Factor=1.2, Factor=1.5)
5. Contrast Modification (Factor=0.7, Factor=1.3, Factor=1.6)
6. Additive Gaussian Noise (Sigma=0.03, Sigma=0.06, Sigma=0.10)
"""

import io
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from app.models_vfm import (
    load_dinov2_small,
    get_dinov2_transforms,
    DINOv2ClassificationHead,
    DINOv2AuthenticityClassifier,
    EMBEDDING_DIM
)
from app.calibration import compute_entropy


def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_downscale_upscale(img: Image.Image, factor: float) -> Image.Image:
    orig_w, orig_h = img.size
    down_w = max(4, int(orig_w * factor))
    down_h = max(4, int(orig_h * factor))
    down = img.resize((down_w, down_h), Image.Resampling.BILINEAR)
    return down.resize((orig_w, orig_h), Image.Resampling.BILINEAR)


def apply_gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness_change(img: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_contrast_change(img: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def apply_additive_gaussian_noise(img: Image.Image, sigma: float, seed: int = 42) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape)
    noisy_arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy_arr * 255.0).astype(np.uint8))


PERTURBATIONS_SPEC = [
    ("Original (Reference)", 0, lambda img: img, "None"),
    ("JPEG Compression", 1, lambda img: apply_jpeg_compression(img, quality=75), "Quality=75"),
    ("JPEG Compression", 2, lambda img: apply_jpeg_compression(img, quality=50), "Quality=50"),
    ("JPEG Compression", 3, lambda img: apply_jpeg_compression(img, quality=25), "Quality=25"),
    ("Downscale-Upscale", 1, lambda img: apply_downscale_upscale(img, factor=0.75), "Factor=0.75"),
    ("Downscale-Upscale", 2, lambda img: apply_downscale_upscale(img, factor=0.50), "Factor=0.50"),
    ("Downscale-Upscale", 3, lambda img: apply_downscale_upscale(img, factor=0.25), "Factor=0.25"),
    ("Gaussian Blur", 1, lambda img: apply_gaussian_blur(img, radius=1.0), "Radius=1.0"),
    ("Gaussian Blur", 2, lambda img: apply_gaussian_blur(img, radius=2.0), "Radius=2.0"),
    ("Gaussian Blur", 3, lambda img: apply_gaussian_blur(img, radius=3.0), "Radius=3.0"),
    ("Brightness Change", 1, lambda img: apply_brightness_change(img, factor=0.80), "Factor=0.80"),
    ("Brightness Change", 2, lambda img: apply_brightness_change(img, factor=1.20), "Factor=1.20"),
    ("Brightness Change", 3, lambda img: apply_brightness_change(img, factor=1.50), "Factor=1.50"),
    ("Contrast Change", 1, lambda img: apply_contrast_change(img, factor=0.70), "Factor=0.70"),
    ("Contrast Change", 2, lambda img: apply_contrast_change(img, factor=1.30), "Factor=1.30"),
    ("Contrast Change", 3, lambda img: apply_contrast_change(img, factor=1.60), "Factor=1.60"),
    ("Additive Gaussian Noise", 1, lambda img: apply_additive_gaussian_noise(img, sigma=0.03), "Sigma=0.03"),
    ("Additive Gaussian Noise", 2, lambda img: apply_additive_gaussian_noise(img, sigma=0.06), "Sigma=0.06"),
    ("Additive Gaussian Noise", 3, lambda img: apply_additive_gaussian_noise(img, sigma=0.10), "Sigma=0.10"),
]


def evaluate_robustness(
    test_paths: list,
    test_labels: np.ndarray,
    model: DINOv2AuthenticityClassifier,
    transform,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    subset_size: int = -1
):
    if subset_size > 0 and subset_size < len(test_paths):
        # Deterministic balanced subset for evaluation
        indices = np.linspace(0, len(test_paths) - 1, subset_size, dtype=int)
        eval_paths = [test_paths[i] for i in indices]
        eval_labels = test_labels[indices]
    else:
        eval_paths = test_paths
        eval_labels = test_labels

    print(f"Loaded {len(eval_paths)} test samples for robustness evaluation...")
    raw_images = [Image.open(p).convert("RGB") for p in eval_paths]

    results_table = []
    baseline_acc = None

    for p_name, severity, transform_fn, param_desc in PERTURBATIONS_SPEC:
        t0 = time.perf_counter()
        transformed_tensors = []
        for img in raw_images:
            t_img = transform_fn(img)
            tensor = transform(t_img)
            transformed_tensors.append(tensor)

        all_logits = []
        with torch.no_grad():
            for b_idx in range(0, len(transformed_tensors), batch_size):
                b_tensors = torch.stack(transformed_tensors[b_idx : b_idx + batch_size]).to(device)
                logits = model(b_tensors)
                all_logits.append(logits.cpu().numpy())

        logits_arr = np.concatenate(all_logits, axis=0)
        exp_l = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
        probs_arr = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

        preds = np.argmax(probs_arr, axis=1)
        confs = np.max(probs_arr, axis=1)
        entropies = compute_entropy(probs_arr)

        acc = accuracy_score(eval_labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(eval_labels, preds, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(eval_labels, probs_arr[:, 1])
        except Exception:
            auc = 0.5

        mean_conf = float(np.mean(confs))
        mean_ent = float(np.mean(entropies))
        elapsed = time.perf_counter() - t0

        if severity == 0:
            baseline_acc = acc
            rel_drop = 0.0
        else:
            rel_drop = (baseline_acc - acc) * 100

        print(f"[{p_name:24} Sev={severity}] Acc: {acc*100:5.2f}% | F1: {f1*100:5.2f}% | AUC: {auc*100:5.2f}% | Drop: {rel_drop:+5.2f}% ({elapsed:.1f}s)")

        results_table.append({
            "Perturbation": p_name,
            "Severity": severity,
            "Parameters": param_desc,
            "Accuracy": acc * 100,
            "Macro_Precision": prec * 100,
            "Macro_Recall": rec * 100,
            "Macro_F1": f1 * 100,
            "ROC_AUC": auc * 100,
            "Mean_Confidence": mean_conf * 100,
            "Mean_Entropy": mean_ent,
            "Accuracy_Drop": rel_drop
        })

    return pd.DataFrame(results_table)
