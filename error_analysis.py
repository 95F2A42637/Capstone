"""
Forensic Error Analysis Module for DINOv2 Authenticity Classifier.

Performs rigorous post-hoc error breakdown over the full held-out test set (N=2,000):
- True Positives, True Negatives, False Positives, False Negatives
- High/Low-Confidence Correct and Incorrect sub-cohorts
- Confusion Matrix and Distributional Calibration Plots
- Deterministic error case extraction without qualitative fabulations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

from app.models_vfm import (
    DINOv2ClassificationHead,
    EMBEDDING_DIM
)
from app.calibration import compute_entropy


def run_test_set_inference(
    features_path: str = "outputs/features/test/features.pt",
    model_path: str = "models/dinov2_authenticity_head.pth",
    calibration_path: str = "models/calibration_params.json",
    device: torch.device = torch.device("cpu")
) -> pd.DataFrame:
    """
    Runs calibrated inference over the full held-out test split representations.
    Returns DataFrame containing per-sample forensic metadata.
    """
    test_data = torch.load(features_path, map_location=device, weights_only=False)
    features = test_data["features"].to(device)
    labels = test_data["labels"].numpy()
    paths = test_data["paths"]
    classes = test_data["classes"]

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hidden_dim", 128)
    head = DINOv2ClassificationHead(
        in_features=EMBEDDING_DIM,
        hidden_dim=hidden_dim,
        num_classes=len(classes)
    ).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    with open(calibration_path, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    temperature = calib_data.get("temperature", 1.0)

    with torch.no_grad():
        raw_logits = head(features)
        calibrated_logits = raw_logits / temperature
        calibrated_probs = F.softmax(calibrated_logits, dim=-1).cpu().numpy()
        raw_logits_np = raw_logits.cpu().numpy()

    preds = np.argmax(calibrated_probs, axis=-1)
    confs = np.max(calibrated_probs, axis=-1)
    entropies = compute_entropy(calibrated_probs)
    is_correct = (preds == labels)

    records = []
    for i in range(len(labels)):
        y_true = labels[i]
        y_pred = preds[i]
        correct = bool(is_correct[i])

        true_name = classes[y_true]
        pred_name = classes[y_pred]

        is_synthetic_true = (true_name == "ai_edited")
        is_synthetic_pred = (pred_name == "ai_edited")

        # Forensic category (Positive = AI-Generated/Synthetic, Negative = Authentic)
        if is_synthetic_true and is_synthetic_pred:
            category = "True Positive (Synthetic)"
        elif (not is_synthetic_true) and (not is_synthetic_pred):
            category = "True Negative (Authentic)"
        elif (not is_synthetic_true) and is_synthetic_pred:
            category = "False Positive (Authentic -> Synthetic)"
        else:
            category = "False Negative (Synthetic -> Authentic)"

        # Confidence sub-cohort
        conf = confs[i]
        if correct and conf >= 0.90:
            conf_tier = "High-Confidence Correct (>=0.90)"
        elif correct and conf <= 0.60:
            conf_tier = "Low-Confidence Correct (<=0.60)"
        elif (not correct) and conf >= 0.90:
            conf_tier = "High-Confidence Incorrect (>=0.90)"
        elif (not correct) and conf <= 0.60:
            conf_tier = "Low-Confidence Incorrect (<=0.60)"
        else:
            conf_tier = "Moderate-Confidence (0.60-0.90)"

        auth_idx = classes.index("authentic")
        synth_idx = classes.index("ai_edited")

        records.append({
            "image_path": paths[i],
            "true_class_idx": int(y_true),
            "true_class_name": true_name,
            "pred_class_idx": int(y_pred),
            "pred_class_name": pred_name,
            "correct": correct,
            "category": category,
            "confidence_tier": conf_tier,
            "calibrated_conf": float(conf),
            "prob_authentic": float(calibrated_probs[i, auth_idx]),
            "prob_synthetic": float(calibrated_probs[i, synth_idx]),
            "raw_logit_authentic": float(raw_logits_np[i, auth_idx]),
            "raw_logit_synthetic": float(raw_logits_np[i, synth_idx]),
            "predictive_entropy": float(entropies[i])
        })

    df = pd.DataFrame(records)
    return df, classes


def compute_aggregate_metrics(df: pd.DataFrame, classes: list) -> dict:
    """Computes comprehensive error metrics and confusion matrix."""
    y_true = df["true_class_idx"].values
    y_pred = df["pred_class_idx"].values
    synth_idx = classes.index("ai_edited")
    y_prob_synth = df["prob_synthetic"].values

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    # Binary ROC-AUC where Positive class is Synthetic (ai_edited)
    binary_true = (df["true_class_name"] == "ai_edited").astype(int)
    auc = roc_auc_score(binary_true, y_prob_synth)

    # Forensic Confusion Matrix
    tp = int(len(df[(df["true_class_name"] == "ai_edited") & (df["pred_class_name"] == "ai_edited")]))
    fn = int(len(df[(df["true_class_name"] == "ai_edited") & (df["pred_class_name"] == "authentic")]))
    fp = int(len(df[(df["true_class_name"] == "authentic") & (df["pred_class_name"] == "ai_edited")]))
    tn = int(len(df[(df["true_class_name"] == "authentic") & (df["pred_class_name"] == "authentic")]))

    total = len(df)
    error_rate = (fp + fn) / total
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    corr_df = df[df["correct"]]
    err_df = df[~df["correct"]]

    return {
        "total_samples": total,
        "classes": classes,
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        },
        "error_rate": float(error_rate),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "correct_stats": {
            "count": int(len(corr_df)),
            "mean_conf": float(corr_df["calibrated_conf"].mean()),
            "std_conf": float(corr_df["calibrated_conf"].std()),
            "mean_entropy": float(corr_df["predictive_entropy"].mean()),
            "std_entropy": float(corr_df["predictive_entropy"].std())
        },
        "incorrect_stats": {
            "count": int(len(err_df)),
            "mean_conf": float(err_df["calibrated_conf"].mean()),
            "std_conf": float(err_df["calibrated_conf"].std()),
            "mean_entropy": float(err_df["predictive_entropy"].mean()),
            "std_entropy": float(err_df["predictive_entropy"].std())
        },
        "incorrect_stats": {
            "count": int(len(err_df)),
            "mean_conf": float(err_df["calibrated_conf"].mean()),
            "std_conf": float(err_df["calibrated_conf"].std()),
            "mean_entropy": float(err_df["predictive_entropy"].mean()),
            "std_entropy": float(err_df["predictive_entropy"].std())
        }
    }


def generate_error_visualizations(df: pd.DataFrame, agg: dict, out_dir: str = "outputs"):
    """Creates confusion matrix and confidence distribution plots."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Confusion Matrix
    cm = np.array([
        [agg["confusion_matrix"]["TN"], agg["confusion_matrix"]["FP"]],
        [agg["confusion_matrix"]["FN"], agg["confusion_matrix"]["TP"]]
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Authentic", "Synthetic (AI)"], fontsize=11)
    ax.set_yticklabels(["Authentic", "Synthetic (AI)"], fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
    ax.set_title("Test Set Confusion Matrix (N=2,000)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "error_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 2. Confidence vs Correctness Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    corr_conf = df[df["correct"]]["calibrated_conf"]
    err_conf = df[~df["correct"]]["calibrated_conf"]

    ax1.hist(corr_conf, bins=25, alpha=0.7, color="teal", label=f"Correct (N={len(corr_conf)})", density=True)
    ax1.hist(err_conf, bins=25, alpha=0.7, color="crimson", label=f"Incorrect (N={len(err_conf)})", density=True)
    ax1.set_xlabel("Calibrated Confidence", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Confidence Distribution: Correct vs Error", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    corr_ent = df[df["correct"]]["predictive_entropy"]
    err_ent = df[~df["correct"]]["predictive_entropy"]

    ax2.hist(corr_ent, bins=25, alpha=0.7, color="teal", label=f"Correct (N={len(corr_ent)})", density=True)
    ax2.hist(err_ent, bins=25, alpha=0.7, color="crimson", label=f"Incorrect (N={len(err_ent)})", density=True)
    ax2.set_xlabel("Predictive Entropy (bits)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Predictive Entropy Distribution", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = os.path.join(out_dir, "error_confidence_vs_correctness.png")
    plt.savefig(dist_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix plot to: {cm_path}")
    print(f"Saved confidence distribution plot to: {dist_path}")
