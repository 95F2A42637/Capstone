"""
High-Performance Modular Feature Caching Pipeline for DINOv2-Small.

This script processes images across the train, val, and test splits using frozen
DINOv2-Small and caches 384-dimensional [CLS] representations to disk.
"""

import os
import time
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image

from app.models_vfm import load_dinov2_small, get_dinov2_transforms, EMBEDDING_DIM


class IndexedImageFolder(Dataset):
    """ImageFolder wrapper that preserves image filepaths alongside sample tensors and labels."""
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = self.dataset.loader(path)
        if self.dataset.transform is not None:
            image = self.dataset.transform(image)
        return image, label, path


def cache_split_features(
    model: torch.nn.Module,
    split_name: str,
    dataset_dir: str,
    output_dir: str,
    batch_size: int = 32,
    max_samples_per_class: int = -1,
    device: torch.device = torch.device("cpu")
):
    split_path = os.path.join(dataset_dir, split_name)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Dataset split not found: {split_path}")

    transform = get_dinov2_transforms()
    full_dataset = IndexedImageFolder(split_path, transform=transform)

    # Optional balanced subsampling for fast experimentation while maintaining strict class balance
    if max_samples_per_class > 0:
        class_indices = {c: [] for c in range(len(full_dataset.classes))}
        for idx, (_, label) in enumerate(full_dataset.dataset.samples):
            if len(class_indices[label]) < max_samples_per_class:
                class_indices[label].append(idx)
        selected_indices = []
        for c in class_indices:
            selected_indices.extend(class_indices[c])
        selected_indices.sort()
        dataset = torch.utils.data.Subset(full_dataset, selected_indices)
    else:
        dataset = full_dataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    split_out_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_out_dir, exist_ok=True)
    cache_file = os.path.join(split_out_dir, "features.pt")

    print(f"\n[{split_name.upper()}] Extracting representations for {len(dataset)} images (Batch Size: {batch_size})...")
    features_list = []
    labels_list = []
    paths_list = []

    t0 = time.perf_counter()
    processed = 0

    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(loader):
            images = images.to(device)
            feat_dict = model.forward_features(images)
            cls_tokens = feat_dict['x_norm_clstoken'].cpu()

            features_list.append(cls_tokens)
            labels_list.append(labels.clone())
            paths_list.extend(paths)

            processed += len(labels)
            if (batch_idx + 1) % 10 == 0 or processed == len(dataset):
                rate = processed / (time.perf_counter() - t0)
                print(f"  Processed {processed}/{len(dataset)} ({rate:.1f} img/s)...", flush=True)

    elapsed_time = time.perf_counter() - t0
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Compute class distribution
    class_counts = {}
    for idx, cname in enumerate(full_dataset.classes):
        class_counts[cname] = int((all_labels == idx).sum().item())

    payload = {
        "features": all_features,
        "labels": all_labels,
        "paths": paths_list,
        "classes": full_dataset.classes,
        "class_to_idx": full_dataset.class_to_idx,
        "class_counts": class_counts,
        "embedding_dim": EMBEDDING_DIM,
        "split": split_name,
        "extraction_time_sec": elapsed_time
    }

    torch.save(payload, cache_file)
    print(f"[{split_name.upper()}] Saved {all_features.shape[0]} representations to: {cache_file}")
    print(f"    Class Counts : {class_counts}")
    print(f"    Extraction Time: {elapsed_time:.2f} s ({len(dataset)/elapsed_time:.1f} img/s)")

    return payload


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Feature Caching Pipeline")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Root dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs/features", help="Output directory for cache")
    parser.add_argument("--batch_size", type=int, default=32, help="Extraction batch size")
    parser.add_argument("--max_train", type=int, default=4000, help="Max samples per class in train split (-1 for all)")
    parser.add_argument("--max_val", type=int, default=1000, help="Max samples per class in val split (-1 for all)")
    parser.add_argument("--max_test", type=int, default=1000, help="Max samples per class in test split (-1 for all)")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Initializing DINOv2 Feature Cache Generator on device: {device}")
    model, metadata = load_dinov2_small(device=device)

    splits_config = [
        ("train", args.max_train),
        ("val", args.max_val),
        ("test", args.max_test)
    ]

    for split_name, max_per_class in splits_config:
        cache_split_features(
            model=model,
            split_name=split_name,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_samples_per_class=max_per_class,
            device=device
        )

    print("\nFeature caching completed across all splits.")


if __name__ == "__main__":
    main()
