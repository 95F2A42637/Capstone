"""
Fine-Grained Patch-Level Explainability for DINOv2 Vision Foundation Model.

This module implements gradient-based patch attribution to produce approximate
spatial model explanations for image authenticity verification decisions.

Pipelines:
1. Input image -> Preprocessing -> DINOv2 blocks 1..11 -> intermediate tokens [1, 257, 384]
2. Autograd tracking on patch tokens -> Block 12 forward -> LayerNorm -> [CLS] token -> Classification Head
3. Backpropagation from target class logit -> Gradient tensor w.r.t. patch tokens [1, 256, 384]
4. L2 norm / gradient-weighted activation reduction -> 256 scalar patch scores -> Reshaped to 16x16
5. Min-max normalization & bicubic interpolation to input resolution (224x224)
6. Generation of raw attribution map, colormapped heatmap, and visual overlay.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from typing import Dict, Tuple, Optional

from app.models_vfm import (
    load_dinov2_small,
    get_dinov2_transforms,
    DINOv2ClassificationHead,
    EMBEDDING_DIM,
    DEFAULT_IMAGE_SIZE
)


class DINOv2PatchAttributor:
    """
    Gradient-based patch attribution engine for DINOv2-Small.
    Computes spatial relevance maps for predicted authenticity classes.
    """
    def __init__(
        self,
        checkpoint_path: str = os.path.join("models", "dinov2_authenticity_head.pth"),
        device: torch.device = torch.device("cpu")
    ):
        self.device = device

        # 1. Load backbone
        self.backbone, _ = load_dinov2_small(device=self.device)
        self.backbone.eval()

        # 2. Load trained classification head
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Classification head checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.classes = ckpt.get("classes", ["ai_edited", "authentic"])
        self.hidden_dim = ckpt.get("hidden_dim", 256)

        self.head = DINOv2ClassificationHead(
            in_features=EMBEDDING_DIM,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.classes)
        ).to(self.device)
        self.head.load_state_dict(ckpt["head_state_dict"])
        self.head.eval()

        self.transform = get_dinov2_transforms(DEFAULT_IMAGE_SIZE)

    def explain(
        self,
        image_input,
        target_class_idx: Optional[int] = None
    ) -> Dict:
        """
        Computes gradient-based patch attribution for an input image.

        Args:
            image_input: Filepath string or PIL Image.
            target_class_idx: Optional target class index (defaults to argmax predicted class).

        Returns:
            Dictionary containing:
                - 'predicted_class': str
                - 'predicted_idx': int
                - 'confidence': float
                - 'probabilities': Dict[str, float]
                - 'attribution_tensor_shape': List[int] [1, 256, 384]
                - 'patch_grid_shape': Tuple[int, int] (16, 16)
                - 'patch_attribution_16x16': np.ndarray of shape (16, 16)
                - 'heatmap_224x224': np.ndarray of shape (224, 224) normalized in [0, 1]
                - 'colored_heatmap': np.ndarray of shape (224, 224, 3) uint8
                - 'overlay_image': PIL.Image.Image
                - 'inference_time_ms': float
                - 'explanation_time_ms': float
        """
        # Load image
        if isinstance(image_input, str):
            orig_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            orig_pil = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        resized_pil = orig_pil.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), Image.Resampling.BILINEAR)
        input_tensor = self.transform(orig_pil).unsqueeze(0).to(self.device)

        t_start = time.perf_counter()

        # Step 1: Forward through backbone blocks 0 to 10
        x_tokens = self.backbone.prepare_tokens_with_masks(input_tensor)
        for blk in self.backbone.blocks[:-1]:
            x_tokens = blk(x_tokens)

        # Enable gradient tracking on intermediate patch representations
        x_tokens = x_tokens.clone().detach().requires_grad_(True)
        x_tokens.retain_grad()

        # Step 2: Forward through final block (block 11) and LayerNorm
        out_tokens = self.backbone.blocks[-1](x_tokens)
        out_norm = self.backbone.norm(out_tokens)

        cls_token = out_norm[:, 0]  # [1, 384]
        patch_tokens = out_norm[:, 1:]  # [1, 256, 384]

        # Step 3: Forward through classification head
        logits = self.head(cls_token)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # Determine target class
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(probs).item())

        predicted_class = self.classes[target_class_idx]
        confidence = float(probs[target_class_idx].item())
        prob_dict = {self.classes[i]: float(probs[i].item()) for i in range(len(self.classes))}

        t_infer = time.perf_counter()

        # Step 4: Backpropagate from selected class logit
        self.head.zero_grad()
        self.backbone.zero_grad()

        target_logit = logits[0, target_class_idx]
        target_logit.backward()

        # Step 5: Extract patch gradients
        # x_tokens: [1, 257, 384] -> index 1: corresponds to 256 patch tokens
        patch_grads = x_tokens.grad[:, 1:, :]  # [1, 256, 384]
        patch_repr = x_tokens[:, 1:, :]        # [1, 256, 384]

        # Gradient-weighted activation attribution (Grad-CAM style dot product per patch)
        # S_p = ReLU( sum_c ( g_{p,c} * a_{p,c} ) )
        grad_act = (patch_grads * patch_repr).sum(dim=-1).squeeze(0)  # [256]
        attribution_scores = torch.clamp(grad_act, min=0.0)

        # Handle zero gradient edge case (fallback to L2 norm of gradients)
        if attribution_scores.max() == 0:
            attribution_scores = torch.norm(patch_grads.squeeze(0), p=2, dim=-1)

        patch_scores_np = attribution_scores.detach().cpu().numpy()

        # Step 6: Reshape 256 scalar scores into 16x16 spatial grid
        attribution_map_16x16 = patch_scores_np.reshape(16, 16)

        # Step 7: Min-Max Normalization of 16x16 map
        min_v = attribution_map_16x16.min()
        max_v = attribution_map_16x16.max()
        if max_v - min_v > 1e-8:
            norm_map_16x16 = (attribution_map_16x16 - min_v) / (max_v - min_v)
        else:
            norm_map_16x16 = np.zeros_like(attribution_map_16x16)

        # Step 8: Bicubic Upsampling to 224x224 input resolution
        norm_map_pil = Image.fromarray((norm_map_16x16 * 255.0).astype(np.uint8))
        upsampled_map = norm_map_pil.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), Image.Resampling.BICUBIC)
        heatmap_224 = np.asarray(upsampled_map, dtype=np.float32) / 255.0

        # Step 9: Render colored heatmap & overlay
        import matplotlib.pyplot as plt
        colormap = plt.get_cmap("jet")
        colored_rgba = colormap(heatmap_224)  # [224, 224, 4]
        colored_rgb = (colored_rgba[:, :, :3] * 255.0).astype(np.uint8)

        # Alpha blend overlay (60% original image, 40% attribution heatmap)
        img_np = np.asarray(resized_pil, dtype=np.float32)
        overlay_np = 0.60 * img_np + 0.40 * colored_rgb.astype(np.float32)
        overlay_np = np.clip(overlay_np, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        t_end = time.perf_counter()

        return {
            "predicted_class": predicted_class,
            "predicted_idx": target_class_idx,
            "confidence": confidence,
            "probabilities": prob_dict,
            "attribution_tensor_shape": list(patch_grads.shape),
            "patch_grid_shape": (16, 16),
            "patch_attribution_16x16": norm_map_16x16,
            "heatmap_224x224": heatmap_224,
            "colored_heatmap": colored_rgb,
            "overlay_image": overlay_pil,
            "original_resized_pil": resized_pil,
            "inference_time_ms": (t_infer - t_start) * 1000,
            "explanation_time_ms": (t_end - t_start) * 1000
        }

    def perturbation_sanity_check(
        self,
        image_input,
        top_k_patches: int = 32
    ) -> Dict[str, float]:
        """
        Attribution Sanity Check via Top-K Patch Perturbation:
        Verifies that zeroing out the top-K highest attribution patches causes a
        significantly larger drop in predicted class logit than zeroing out random patches.
        """
        if isinstance(image_input, str):
            orig_pil = Image.open(image_input).convert("RGB")
        else:
            orig_pil = image_input.convert("RGB")

        # 1. Base prediction
        expl = self.explain(orig_pil)
        pred_idx = expl["predicted_idx"]
        norm_map = expl["patch_attribution_16x16"].flatten()

        top_indices = np.argsort(norm_map)[::-1][:top_k_patches]
        np.random.seed(42)
        random_indices = np.random.choice(len(norm_map), size=top_k_patches, replace=False)

        # Baseline logit
        input_t = self.transform(orig_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.backbone.forward_features(input_t)
            base_logits = self.head(feat['x_norm_clstoken'])
            base_target_logit = float(base_logits[0, pred_idx].item())

        # Perturbation function (mask out 14x14 patches on input image)
        def mask_image_patches(indices):
            masked = orig_pil.resize((224, 224), Image.Resampling.BILINEAR).copy()
            arr = np.array(masked)
            for idx in indices:
                py = (idx // 16) * 14
                px = (idx % 16) * 14
                arr[py:py+14, px:px+14, :] = 128  # neutral gray replacement
            masked_pil = Image.fromarray(arr)
            masked_t = self.transform(masked_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat_m = self.backbone.forward_features(masked_t)
                logits_m = self.head(feat_m['x_norm_clstoken'])
                return float(logits_m[0, pred_idx].item())

        logit_after_top_mask = mask_image_patches(top_indices)
        logit_after_rand_mask = mask_image_patches(random_indices)

        drop_top = base_target_logit - logit_after_top_mask
        drop_random = base_target_logit - logit_after_rand_mask

        return {
            "base_logit": base_target_logit,
            "logit_after_top_mask": logit_after_top_mask,
            "logit_after_rand_mask": logit_after_rand_mask,
            "drop_top_patches": drop_top,
            "drop_random_patches": drop_random,
            "top_drop_ratio": drop_top / (drop_random + 1e-6)
        }
