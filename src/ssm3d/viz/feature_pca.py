"""Feature-map PCA visualization.

Takes a (T, C) or (H, W, C) feature tensor from the backbone, reduces to 3
channels via PCA, min-max normalizes per image, and saves a PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def feature_pca_image(
    feat: torch.Tensor,
    spatial_hw: tuple[int, int] | None = None,
    upsample_to: tuple[int, int] | None = None,
) -> np.ndarray:
    """Reduce a feature map to an RGB uint8 image.

    Args:
        feat: (H, W, C) or (N, C) where N = H*W
        spatial_hw: (H, W) when feat is flat (N, C); ignored otherwise
        upsample_to: optional (H_out, W_out) to resize to after PCA (bilinear).
            Useful to save a visible PNG from a tiny patch grid; PCA output is
            continuous so bilinear avoids the block-quantization of nearest.

    Returns:
        (H', W', 3) uint8 array suitable for Image.fromarray.
    """
    f = feat.detach().cpu().float()
    if f.ndim == 2:
        assert spatial_hw is not None, "pass spatial_hw for (N, C) input"
        H, W = spatial_hw
        f = f.view(H, W, -1)
    H, W, C = f.shape
    X = f.reshape(-1, C).numpy()  # (N, C)

    X_centered = X - X.mean(axis=0, keepdims=True)
    _U, _S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    top3 = X_centered @ Vt[:3].T  # (N, 3)

    lo = top3.min(axis=0, keepdims=True)
    hi = top3.max(axis=0, keepdims=True)
    top3 = (top3 - lo) / np.maximum(hi - lo, 1e-8)
    img = (top3 * 255).clip(0, 255).astype(np.uint8).reshape(H, W, 3)

    if upsample_to is not None:
        H_out, W_out = upsample_to
        img = np.asarray(Image.fromarray(img).resize((W_out, H_out), Image.BILINEAR))
    return img


def save_feature_pca(
    feat: torch.Tensor,
    path: Path,
    spatial_hw: tuple[int, int] | None = None,
    upsample_to: tuple[int, int] | None = None,
) -> Path:
    img = feature_pca_image(feat, spatial_hw=spatial_hw, upsample_to=upsample_to)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path
