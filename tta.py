"""
Test-Time Augmentation (TTA) Evaluation Module.

Evaluates test-time augmentation (horizontal flip) vs standard single-crop inference
using probability averaging on the held-out test split.
"""

import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from app.calibration import compute_entropy
from app.models_vfm import DINOv2AuthenticityClassifier


def evaluate_tta(
    test_paths: list,
    test_labels: np.ndarray,
    model: DINOv2AuthenticityClassifier,
    transform,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu")
):
    print(f"Executing TTA evaluation on {len(test_paths)} test samples...")
    raw_images = [Image.open(p).convert("RGB") for p in test_paths]

    # --- 1. Standard Single-View Inference ---
    t0_single = time.perf_counter()
    single_tensors = [transform(img) for img in raw_images]
    single_logits = []
    with torch.no_grad():
        for b_idx in range(0, len(single_tensors), batch_size):
            b_t = torch.stack(single_tensors[b_idx : b_idx + batch_size]).to(device)
            single_logits.append(model(b_t).cpu().numpy())
    time_single = time.perf_counter() - t0_single

    s_logits_arr = np.concatenate(single_logits, axis=0)
    exp_s = np.exp(s_logits_arr - np.max(s_logits_arr, axis=-1, keepdims=True))
    single_probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    s_preds = np.argmax(single_probs, axis=1)
    s_acc = accuracy_score(test_labels, s_preds)
    s_prec, s_rec, s_f1, _ = precision_recall_fscore_support(test_labels, s_preds, average='macro', zero_division=0)
    s_auc = roc_auc_score(test_labels, single_probs[:, 1])
    s_conf = float(np.mean(np.max(single_probs, axis=1)))
    s_ent = float(np.mean(compute_entropy(single_probs)))

    # --- 2. Test-Time Augmentation (Original + Horizontal Flip) ---
    t0_tta = time.perf_counter()
    flip_images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in raw_images]
    flip_tensors = [transform(img) for img in flip_images]
    flip_logits = []
    with torch.no_grad():
        for b_idx in range(0, len(flip_tensors), batch_size):
            b_t = torch.stack(flip_tensors[b_idx : b_idx + batch_size]).to(device)
            flip_logits.append(model(b_t).cpu().numpy())
    time_tta = (time.perf_counter() - t0_tta) + time_single  # total time for both views

    f_logits_arr = np.concatenate(flip_logits, axis=0)
    exp_f = np.exp(f_logits_arr - np.max(f_logits_arr, axis=-1, keepdims=True))
    flip_probs = exp_f / np.sum(exp_f, axis=-1, keepdims=True)

    # Probability averaging across 2 views
    tta_probs = 0.5 * (single_probs + flip_probs)
    tta_preds = np.argmax(tta_probs, axis=1)
    tta_acc = accuracy_score(test_labels, tta_preds)
    tta_prec, tta_rec, tta_f1, _ = precision_recall_fscore_support(test_labels, tta_preds, average='macro', zero_division=0)
    tta_auc = roc_auc_score(test_labels, tta_probs[:, 1])
    tta_conf = float(np.mean(np.max(tta_probs, axis=1)))
    tta_ent = float(np.mean(compute_entropy(tta_probs)))

    overhead = (time_tta / time_single)

    res_df = pd.DataFrame([
        {
            "Strategy": "Standard Single-Crop",
            "Accuracy": s_acc * 100,
            "Macro_Precision": s_prec * 100,
            "Macro_Recall": s_rec * 100,
            "Macro_F1": s_f1 * 100,
            "ROC_AUC": s_auc * 100,
            "Mean_Confidence": s_conf * 100,
            "Mean_Entropy": s_ent,
            "Total_Time_Sec": time_single,
            "Inference_Overhead": "1.00x"
        },
        {
            "Strategy": "TTA (Original + H-Flip)",
            "Accuracy": tta_acc * 100,
            "Macro_Precision": tta_prec * 100,
            "Macro_Recall": tta_rec * 100,
            "Macro_F1": tta_f1 * 100,
            "ROC_AUC": tta_auc * 100,
            "Mean_Confidence": tta_conf * 100,
            "Mean_Entropy": tta_ent,
            "Total_Time_Sec": time_tta,
            "Inference_Overhead": f"{overhead:.2f}x"
        }
    ])
    return res_df
