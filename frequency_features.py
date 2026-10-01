"""
Compact Frequency-Domain Feature Extraction via 2D Fast Fourier Transform (FFT).

This module extracts a compact, mathematically rigorous 32-dimensional spectral
feature vector from an input image to analyze frequency-domain footprints without
requiring large 2D frequency map inputs into the classifier.

Feature Composition (32 Dimensions):
1. Radial Band Energy Proportions (8 dimensions):
   - Energy concentrated in 8 concentric radial frequency annuli from DC center to Nyquist edge.
2. High / Low Energy Ratios (4 dimensions):
   - Direct energy ratios between high-frequency and low-frequency bands.
3. Global Spectral Statistical Moments (8 dimensions):
   - Mean, variance, skewness, kurtosis, standard deviation, max, median, and dynamic range of log magnitude.
4. Directional / Angular Energy Distribution (8 dimensions):
   - Energy partitioned across 8 azimuthal/angular sectors (0 to 180 degrees) to capture directional synthesis artifacts.
5. Information Theoretic & Spectral Shape Descriptors (4 dimensions):
   - Spectral Flatness (Wiener entropy: geometric mean / arithmetic mean).
   - Spectral Entropy (normalized Shannon entropy of the power distribution).
   - High-Frequency Spectral Roll-off (85th and 95th percentile energy thresholds).
"""

import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from typing import Union, List, Dict


FREQ_FEATURE_DIM = 32


def extract_compact_fft_features(image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Extracts a 32-dimensional frequency representation from an input image.

    Args:
        image_input: Filepath string, PIL Image, or numpy array.

    Returns:
        1D numpy array of shape (32,) with normalized spectral statistics.
    """
    # 1. Load and convert to grayscale float array in [0, 1]
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            img = Image.fromarray(image_input).convert("L")
        else:
            img = Image.fromarray(image_input)
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # Standardize spatial resolution to 224x224 for consistent frequency bins
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float64) / 255.0

    # 2. 2D Fast Fourier Transform & Shift DC to center
    fft2d = np.fft.fft2(arr)
    fft_shifted = np.fft.fftshift(fft2d)

    # Magnitude and Power Spectrum
    mag = np.abs(fft_shifted)
    power = mag ** 2
    eps = 1e-10
    log_mag = np.log1p(mag)  # log(1 + |F|)

    H, W = arr.shape
    cy, cx = H // 2, W // 2

    # Coordinate grids
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)

    total_power = np.sum(power) + eps

    # --- Feature Group 1: Radial Band Energies (8 bins) ---
    num_radial_bands = 8
    radial_energies = []
    r_step = max_r / num_radial_bands
    for i in range(num_radial_bands):
        mask = (r >= i * r_step) & (r < (i + 1) * r_step)
        band_power = np.sum(power[mask]) / total_power
        radial_energies.append(band_power)
    radial_energies = np.array(radial_energies, dtype=np.float64)

    # --- Feature Group 2: High / Low Frequency Energy Ratios (4 bins) ---
    low_band = radial_energies[0] + radial_energies[1] + eps
    mid_low = radial_energies[2] + radial_energies[3] + eps
    mid_high = radial_energies[4] + radial_energies[5] + eps
    high_band = radial_energies[6] + radial_energies[7] + eps

    ratio_high_low = high_band / low_band
    ratio_high_mid = high_band / (mid_low + mid_high)
    ratio_mid_low = (mid_low + mid_high) / low_band
    ratio_extreme = high_band / (total_power + eps)
    energy_ratios = np.array([ratio_high_low, ratio_high_mid, ratio_mid_low, ratio_extreme], dtype=np.float64)

    # --- Feature Group 3: Global Log-Magnitude Statistical Moments (8 bins) ---
    log_vals = log_mag.flatten()
    mean_val = np.mean(log_vals)
    std_val = np.std(log_vals)
    var_val = np.var(log_vals)
    median_val = np.median(log_vals)
    max_val = np.max(log_vals)
    dyn_range = max_val - np.min(log_vals)
    
    # Standardized 3rd (skewness) and 4th (kurtosis) central moments
    if std_val > eps:
        skew_val = np.mean(((log_vals - mean_val) / std_val) ** 3)
        kurt_val = np.mean(((log_vals - mean_val) / std_val) ** 4) - 3.0  # excess kurtosis
    else:
        skew_val = 0.0
        kurt_val = 0.0

    stat_moments = np.array([
        mean_val, std_val, var_val, median_val,
        max_val, dyn_range, skew_val, kurt_val
    ], dtype=np.float64)

    # --- Feature Group 4: Directional / Azimuthal Sector Distribution (8 bins) ---
    # Angles relative to center (0 to pi)
    theta = np.arctan2(y - cy, x - cx) % np.pi
    num_sectors = 8
    sector_step = np.pi / num_sectors
    sector_energies = []
    for s in range(num_sectors):
        mask = (theta >= s * sector_step) & (theta < (s + 1) * sector_step)
        s_power = np.sum(power[mask]) / total_power
        sector_energies.append(s_power)
    sector_energies = np.array(sector_energies, dtype=np.float64)

    # --- Feature Group 5: Spectral Shape & Information Entropy (4 bins) ---
    # 1. Spectral Flatness (Wiener entropy)
    pos_power = power[power > eps]
    if len(pos_power) > 0:
        geom_mean = np.exp(np.mean(np.log(pos_power)))
        arith_mean = np.mean(pos_power)
        spectral_flatness = float(geom_mean / (arith_mean + eps))
    else:
        spectral_flatness = 0.0

    # 2. Spectral Entropy (normalized Shannon entropy)
    prob_dist = power / total_power
    pos_probs = prob_dist[prob_dist > eps]
    spectral_entropy = -float(np.sum(pos_probs * np.log2(pos_probs))) / np.log2(H * W)

    # 3. Spectral Roll-off (frequency radius where 85% and 95% of total power is concentrated)
    sorted_indices = np.argsort(r.flatten())
    sorted_r = r.flatten()[sorted_indices]
    cum_power = np.cumsum(power.flatten()[sorted_indices]) / total_power
    rolloff_85 = float(sorted_r[np.searchsorted(cum_power, 0.85)] / max_r)
    rolloff_95 = float(sorted_r[np.searchsorted(cum_power, 0.95)] / max_r)

    spectral_shape = np.array([spectral_flatness, spectral_entropy, rolloff_85, rolloff_95], dtype=np.float64)

    # Concatenate all groups into 32-D vector
    feature_vector = np.concatenate([
        radial_energies,      # 8
        energy_ratios,        # 4
        stat_moments,         # 8
        sector_energies,      # 8
        spectral_shape        # 4
    ]).astype(np.float32)

    # Clean any potential numerical artifacts
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)

    assert len(feature_vector) == FREQ_FEATURE_DIM, f"Expected {FREQ_FEATURE_DIM}, got {len(feature_vector)}"
    return feature_vector


class FrequencyFusionMLP(nn.Module):
    """
    Lightweight Multimodal Fusion Head combining DINOv2 CLS (384-D) and FFT statistics (32-D).
    
    Architecture:
    DINOv2 CLS (384) -> LayerNorm -> Linear(384 -> 256) -> GELU ──┐
                                                                 ├── Concatenate (320) -> Dropout(0.2) -> Linear(320 -> 2)
    FFT Vector (32)  -> BatchNorm -> Linear(32 -> 64)   -> GELU ──┘
    """
    def __init__(self, visual_dim: int = 384, freq_dim: int = 32, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        # Visual branch
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.visual_fc = nn.Linear(visual_dim, 256)
        self.visual_act = nn.GELU()

        # Frequency branch
        self.freq_norm = nn.BatchNorm1d(freq_dim)
        self.freq_fc = nn.Linear(freq_dim, 64)
        self.freq_act = nn.GELU()

        # Joint fusion
        self.dropout = nn.Dropout(dropout)
        self.fusion_fc = nn.Linear(256 + 64, num_classes)

    def forward(self, v_feat: torch.Tensor, f_feat: torch.Tensor) -> torch.Tensor:
        v = self.visual_act(self.visual_fc(self.visual_norm(v_feat)))
        f = self.freq_act(self.freq_fc(self.freq_norm(f_feat)))
        joint = torch.cat([v, f], dim=-1)
        joint = self.dropout(joint)
        logits = self.fusion_fc(joint)
        return logits
