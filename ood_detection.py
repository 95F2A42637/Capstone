"""
Lightweight Embedding Distance-Based Out-of-Distribution (OOD) Indicator.

Algorithm:
Fits class-conditional centroids mu_c in the 384-dimensional DINOv2 embedding space
using the in-distribution TRAINING set only:
    mu_c = (1 / N_c) * sum_{i in c} z_i

For a query embedding z:
    OOD_Score(z) = min_c || z - mu_c ||_2 (Minimum Euclidean distance to in-distribution class centroids)

A sample with OOD_Score above threshold gamma is flagged as out-of-distribution.
The threshold gamma is determined using the 95th percentile of validation in-distribution distances.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, Tuple, List


class DINOv2CentroidOODDetector:
    """
    Class-centroid Euclidean distance detector in DINOv2 representation space.
    Fitted using training representations only.
    """
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.centroids = {}
        self.classes = []
        self.threshold_95 = None
        self.threshold_99 = None

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray, classes: List[str]):
        self.classes = classes
        self.centroids = {}

        for c_idx, c_name in enumerate(classes):
            mask = (train_labels == c_idx)
            c_feats = train_features[mask]
            self.centroids[c_name] = np.mean(c_feats, axis=0)

        print(f"Fitted {len(self.centroids)} class centroids in {train_features.shape[1]}-D DINOv2 space (Metric: {self.metric}).")

    def compute_ood_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Computes the minimum distance from each sample to any in-distribution class centroid:
        If metric == 'cosine':
            score(z) = min_c (1 - (z . mu_c) / (||z|| * ||mu_c||))
        If metric == 'euclidean':
            score(z) = min_c || z - mu_c ||_2
        """
        distances = []
        for c_name in self.classes:
            mu = self.centroids[c_name]
            if self.metric == "cosine":
                feat_norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
                mu_norm = np.linalg.norm(mu) + 1e-12
                cos_sim = np.dot(features, mu) / (feat_norm.squeeze(-1) * mu_norm)
                d = 1.0 - cos_sim
                distances.append(d[:, np.newaxis])
            else:
                d = np.linalg.norm(features - mu, axis=1, keepdims=True)
                distances.append(d)

        all_dists = np.hstack(distances)
        min_dists = np.min(all_dists, axis=1)
        return min_dists

    def set_threshold_from_val(self, val_features: np.ndarray):
        val_scores = self.compute_ood_scores(val_features)
        self.threshold_95 = float(np.percentile(val_scores, 95))
        self.threshold_99 = float(np.percentile(val_scores, 99))
        print(f"Calibration on validation in-distribution set:")
        print(f"    OOD Threshold (95th percentile) : {self.threshold_95:.4f}")
        print(f"    OOD Threshold (99th percentile) : {self.threshold_99:.4f}")
        return self.threshold_95

    def predict_ood(self, features: np.ndarray, threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.compute_ood_scores(features)
        th = threshold if threshold is not None else self.threshold_95
        is_ood = (scores > th)
        return is_ood, scores
