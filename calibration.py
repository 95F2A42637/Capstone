"""
Post-Hoc Confidence Calibration & Uncertainty Quantification for DINOv2.

Implements:
1. Temperature Scaling optimization on validation logits only.
2. Expected Calibration Error (ECE) calculation with equal-mass / equal-width binning.
3. Negative Log-Likelihood (NLL) and Brier Score evaluation.
4. Reliability Diagram plotting (before and after calibration).
5. Predictive Entropy calculation as an uncertainty indicator:
   H(p) = -sum(p_i * log(p_i))
6. Confidence distribution analysis for correct vs. incorrect predictions.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict, Tuple, List, Optional


class TemperatureScaler:
    """
    Fits a single positive scalar temperature parameter T on validation logits
    to calibrate softmax probabilities via post-hoc temperature scaling:
        p_calibrated = softmax(logits / T)
    """
    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Optimizes scalar T > 0 on validation set using Negative Log-Likelihood objective via L-BFGS.
        """
        t_logits = torch.tensor(val_logits, dtype=torch.float32)
        t_labels = torch.tensor(val_labels, dtype=torch.long)
        nll_criterion = nn.CrossEntropyLoss()

        log_temperature = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.01, max_iter=100)

        def eval_loss():
            optimizer.zero_grad()
            t = torch.exp(log_temperature)
            loss = nll_criterion(t_logits / t, t_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperature = float(torch.exp(log_temperature).item())
        return self.temperature

    def scale_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.scale_logits(logits)
        # Numerically stable softmax
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Tuple[float, Dict]:
    """
    Computes Expected Calibration Error (ECE) and returns reliability diagram statistics.
    ECE = sum_b (|acc(b) - conf(b)| * |b| / N)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_counts.append(0)

    diag_data = {
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist()
    }
    return float(ece), diag_data


def compute_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Computes Negative Log-Likelihood (Cross-Entropy)."""
    clipped = np.clip(probs, eps, 1.0 - eps)
    n = len(labels)
    # Binary cross entropy / multi-class cross entropy
    correct_probs = clipped[np.arange(n), labels]
    return float(-np.mean(np.log(correct_probs)))


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Multi-Class Brier Score: (1/N) * sum_i sum_k (p_ik - y_ik)^2"""
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes Shannon Predictive Entropy as a normalized uncertainty indicator:
    H(p) = -sum(p_i * log2(p_i))
    Normalized to range [0, 1] for binary classification.
    """
    clipped = np.clip(probs, eps, 1.0)
    entropy_nats = -np.sum(clipped * np.log(clipped), axis=1)
    entropy_bits = entropy_nats / np.log(2.0)  # max is 1.0 for binary
    return entropy_bits


def plot_reliability_diagram(diag_data: Dict, ece: float, title: str, save_path: str):
    """Renders a clean reliability diagram bar plot with gap visualization."""
    bin_accs = np.array(diag_data["bin_accs"])
    bin_confs = np.array(diag_data["bin_confs"])
    bin_counts = np.array(diag_data["bin_counts"])
    n_bins = len(bin_accs)
    centers = np.linspace(0.5 / n_bins, 1.0 - 0.5 / n_bins, n_bins)
    width = 1.0 / n_bins

    plt.figure(figsize=(5.5, 5))
    # Perfect calibration diagonal
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration", alpha=0.7)

    # Actual accuracy bars
    plt.bar(centers, bin_accs, width=width * 0.85, alpha=0.7, color='steelblue', label="Model Accuracy", edgecolor='black')

    # Calibration gap (difference between confidence and accuracy)
    for c, acc, conf, cnt in zip(centers, bin_accs, bin_confs, bin_counts):
        if cnt > 0:
            if conf > acc:
                plt.bar(c, conf - acc, bottom=acc, width=width * 0.85, alpha=0.3, color='salmon', hatch='//')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Confidence (Softmax Probability)")
    plt.ylabel("Observed Empirical Accuracy")
    plt.title(f"{title}\nECE: {ece * 100:.2f}%")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
