"""
DINOv2-Small Single-Image Inference CLI & API.

Loads the frozen DINOv2 backbone and the Phase 2 trained classification head
checkpoint (models/dinov2_authenticity_head.pth) to perform CPU inference on any test image.
"""

import os
import argparse
import torch
from PIL import Image

from app.models_vfm import (
    load_dinov2_small,
    get_dinov2_transforms,
    DINOv2ClassificationHead,
    DINOv2AuthenticityClassifier,
    EMBEDDING_DIM
)


def load_authenticity_pipeline(
    checkpoint_path: str = os.path.join("models", "dinov2_authenticity_head.pth"),
    device: torch.device = torch.device("cpu")
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. Run app/train_vfm.py first."
        )

    # 1. Load frozen DINOv2 backbone
    backbone, _ = load_dinov2_small(device=device)

    # 2. Load head weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = checkpoint.get("hidden_dim", 256)
    classes = checkpoint.get("classes", ["authentic", "ai_edited"])

    head = DINOv2ClassificationHead(
        in_features=EMBEDDING_DIM,
        hidden_dim=hidden_dim,
        num_classes=len(classes)
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    # 3. Assemble End-to-End model
    full_model = DINOv2AuthenticityClassifier(backbone, head).to(device)
    full_model.eval()

    transforms = get_dinov2_transforms()

    return full_model, classes, transforms


def predict_single_image(
    image_path: str,
    pipeline=None,
    device: torch.device = torch.device("cpu")
):
    if pipeline is None:
        model, classes, transform = load_authenticity_pipeline(device=device)
    else:
        model, classes, transform = pipeline

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    raw_image = Image.open(image_path).convert("RGB")
    tensor = transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)
        predicted_idx = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_idx].item())

    predicted_class = classes[predicted_idx]
    prob_dict = {classes[i]: float(probabilities[i].item()) for i in range(len(classes))}

    return {
        "image_path": image_path,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Image Authenticity with DINOv2-Small")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="models/dinov2_authenticity_head.pth")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Loading DINOv2 authenticity classifier on {device}...")
    pipeline = load_authenticity_pipeline(args.checkpoint, device=device)

    res = predict_single_image(args.image, pipeline=pipeline, device=device)
    print("\n" + "=" * 50)
    print("DINOv2 AUTHENTICITY VERIFICATION RESULT")
    print("=" * 50)
    print(f"Image       : {res['image_path']}")
    print(f"Prediction  : {res['predicted_class'].upper()}")
    print(f"Confidence  : {res['confidence']*100:.2f}%")
    print("Probabilities:")
    for c, p in res["probabilities"].items():
        print(f"  - {c:>12}: {p*100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
