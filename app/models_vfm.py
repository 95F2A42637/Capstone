"""
DINOv2-Small Foundation Model Loader Module for Image Authenticity Verification.

This module provides a CPU-friendly, robust loader for the facebook/dinov2-small
(ViT-S/14) Vision Foundation Model using local, offline cached weights.
"""

import os
import torch
from torchvision import transforms
from typing import Dict, Tuple, Optional

# Constants
EMBEDDING_DIM = 384
PATCH_SIZE = 14
DEFAULT_IMAGE_SIZE = 224
EXPECTED_PATCH_GRID = (16, 16)
EXPECTED_PATCH_TOKENS = 256

# Candidate local weight locations
DEFAULT_LOCAL_WEIGHTS = [
    os.path.join("models", "dinov2_vits14_pretrain.pth"),
    r"C:\Users\ADMIN\OneDrive\Documents\capstone\models\dinov2_vits14_pretrain.pth",
]

# Candidate local torch.hub repository directories
DEFAULT_LOCAL_REPOS = [
    os.path.join("models", "dinov2_repo"),
    os.path.expanduser(r"~/.cache/torch/hub/facebookresearch_dinov2_main"),
    r"C:\Users\ADMIN\OneDrive\Documents\capstone\models\dinov2_repo",
]


def get_dinov2_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Standard normalization and resizing pipeline expected by DINOv2.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def find_existing_path(candidates: list) -> Optional[str]:
    """Return the first candidate path that exists on disk."""
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def load_dinov2_small(
    weights_path: Optional[str] = None,
    repo_dir: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Dict[str, str]]:
    """
    Loads DINOv2-Small (dinov2_vits14) in pure offline mode without requiring internet access.
    
    Args:
        weights_path: Path to the .pth pretrained weight file.
        repo_dir: Path to the offline torch.hub repository directory for DINOv2.
        device: Target execution device (defaults to cpu).
        
    Returns:
        Tuple containing the instantiated PyTorch model in eval mode and a metadata dictionary.
    """
    if device is None:
        device = torch.device("cpu")

    # Locate weights
    selected_weights = weights_path or find_existing_path(DEFAULT_LOCAL_WEIGHTS)
    if selected_weights is None or not os.path.isfile(selected_weights):
        raise FileNotFoundError(
            f"Pretrained DINOv2 weights not found. Searched: {DEFAULT_LOCAL_WEIGHTS}"
        )

    # Locate repo
    selected_repo = repo_dir or find_existing_path(DEFAULT_LOCAL_REPOS)
    if selected_repo is None or not os.path.isdir(selected_repo):
        raise FileNotFoundError(
            f"DINOv2 source repo not found. Searched: {DEFAULT_LOCAL_REPOS}"
        )

    # Instantiate model from local repo
    model = torch.hub.load(
        selected_repo,
        "dinov2_vits14",
        source="local",
        pretrained=False
    )

    # Load weights
    state_dict = torch.load(selected_weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Freeze parameters by default for feature extractor usage
    for param in model.parameters():
        param.requires_grad = False

    metadata = {
        "model_name": "DINOv2-Small (ViT-S/14)",
        "source_repo": selected_repo,
        "weights_path": selected_weights,
        "device": str(device),
        "embedding_dim": str(EMBEDDING_DIM),
        "patch_size": str(PATCH_SIZE),
        "expected_patch_tokens": str(EXPECTED_PATCH_TOKENS),
        "offline": "True"
    }

    return model, metadata


def extract_features(
    model: torch.nn.Module,
    x: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Forward pass extracting both [CLS] representation and patch tokens.
    
    Args:
        model: DINOv2 vision transformer instance.
        x: Preprocessed image tensor [B, 3, 224, 224].
        
    Returns:
        Dictionary containing:
            - 'cls_token': Tensor [B, 384]
            - 'patch_tokens': Tensor [B, 256, 384]
            - 'combined_avg': Tensor [B, 384] (mean across spatial patch tokens)
    """
    with torch.no_grad():
        feat_dict = model.forward_features(x)
        cls_token = feat_dict['x_norm_clstoken']
        patch_tokens = feat_dict['x_norm_patchtokens']
        combined_avg = patch_tokens.mean(dim=1)

    return {
        "cls_token": cls_token,
        "patch_tokens": patch_tokens,
        "combined_avg": combined_avg
    }


class DINOv2ClassificationHead(torch.nn.Module):
    """
    Lightweight MLP head for binary authenticity classification.
    384-D CLS embedding -> LayerNorm -> Linear(384 -> 256) -> GELU -> Dropout -> Linear(256 -> 2)
    """
    def __init__(self, in_features: int = EMBEDDING_DIM, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_features)
        self.fc1 = torch.nn.Linear(in_features, hidden_dim)
        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


class DINOv2AuthenticityClassifier(torch.nn.Module):
    """
    End-to-End model combining frozen DINOv2 backbone with trainable classification head.
    """
    def __init__(self, backbone: torch.nn.Module, head: DINOv2ClassificationHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

        # Keep backbone parameters frozen
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_dict = self.backbone.forward_features(x)
        cls_token = feat_dict['x_norm_clstoken']
        logits = self.head(cls_token)
        return logits
