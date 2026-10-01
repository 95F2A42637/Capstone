"""
Extraction and caching script for 32-D frequency-domain FFT features.

Reads image paths aligned with outputs/features/ splits and extracts compact
numerical frequency features into outputs/features_frequency/.
"""

import os
import time
import argparse
import torch
import numpy as np

from app.frequency_features import extract_compact_fft_features, FREQ_FEATURE_DIM


def cache_frequency_split(split_name: str, dinov2_cache_dir: str, output_freq_dir: str):
    dinov2_file = os.path.join(dinov2_cache_dir, split_name, "features.pt")
    if not os.path.exists(dinov2_file):
        raise FileNotFoundError(f"DINOv2 cache file not found: {dinov2_file}")

    print(f"\n[{split_name.upper()}] Loading image paths from DINOv2 cache...")
    d_data = torch.load(dinov2_file, weights_only=False)
    paths = d_data["paths"]
    labels = d_data["labels"]
    classes = d_data["classes"]
    num_samples = len(paths)

    print(f"[{split_name.upper()}] Extracting 32-D FFT descriptors for {num_samples} samples...")
    t0 = time.perf_counter()
    freq_vectors = []

    for idx, img_path in enumerate(paths):
        vec = extract_compact_fft_features(img_path)
        freq_vectors.append(vec)

        if (idx + 1) % 500 == 0 or (idx + 1) == num_samples:
            rate = (idx + 1) / (time.perf_counter() - t0)
            print(f"  Processed {idx + 1}/{num_samples} ({rate:.1f} img/s)...", flush=True)

    elapsed_time = time.perf_counter() - t0
    freq_tensor = torch.tensor(np.array(freq_vectors), dtype=torch.float32)

    # Save payload
    split_out = os.path.join(output_freq_dir, split_name)
    os.makedirs(split_out, exist_ok=True)
    out_file = os.path.join(split_out, "features_frequency.pt")

    payload = {
        "features": freq_tensor,
        "labels": labels,
        "paths": paths,
        "classes": classes,
        "freq_dim": FREQ_FEATURE_DIM,
        "split": split_name,
        "extraction_time_sec": elapsed_time
    }
    torch.save(payload, out_file)
    print(f"[{split_name.upper()}] Saved {freq_tensor.shape[0]} FFT representations to: {out_file}")
    print(f"    Tensor Shape   : {list(freq_tensor.shape)}")
    print(f"    Extraction Time: {elapsed_time:.2f} s ({num_samples / elapsed_time:.1f} img/s)")

    return payload


def main():
    parser = argparse.ArgumentParser(description="Extract and cache 32-D FFT frequency features")
    parser.add_argument("--dinov2_cache_dir", type=str, default="outputs/features")
    parser.add_argument("--output_freq_dir", type=str, default="outputs/features_frequency")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 3: EXTRACTING & CACHING COMPACT FREQUENCY FEATURES (2D FFT)")
    print("=" * 70)

    splits = ["train", "val", "test"]
    total_start = time.perf_counter()
    for s in splits:
        cache_frequency_split(s, args.dinov2_cache_dir, args.output_freq_dir)

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print(f"All frequency features cached successfully in {total_time:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
