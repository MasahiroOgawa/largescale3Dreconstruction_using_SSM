"""End-to-end visualization test.

Runs a minimal version of scripts/run_demo.py on synthetic data and asserts
all four expected PNG artifacts are produced and non-trivial.

We deliberately use synthetic inputs (and fake COCO samples) so this test
doesn't need network access. The real demo with ETH3D + COCO-mini lives in
scripts/run_demo.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from ssm3d.mamba3 import Mamba3CrossAttention, RoPE2D
from ssm3d.model import SSM3DNet
from ssm3d.viz import (
    TinyInstanceSegHead,
    save_cross_attention_heatmap,
    save_depth_colormap,
    save_feature_pca,
    save_seg_overlay,
    train_seg_head,
)


def _is_non_trivial(path: Path) -> bool:
    """Loaded image must have pixel variance — not a constant color."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return float(arr.std()) > 1.0


def test_four_visuals_produced(tmp_path: Path):
    torch.manual_seed(0)
    img_size, patch_size = 112, 16
    net = SSM3DNet(size="small", img_size=img_size, patch_size=patch_size, depth=2, head_hidden=32)
    net.eval()

    # Two random "views".
    images = torch.rand(2, 3, img_size, img_size)

    with torch.no_grad():
        out = net(images.unsqueeze(0))
    feats = out["features"][0]  # (2, N, C)
    depth = out["depth"][0]
    h, w = out["grid_hw"]
    C = feats.shape[-1]

    # 1) feature_pca
    feature_pca_path = save_feature_pca(feats[0], tmp_path / "feature_pca_view0.png", spatial_hw=(h, w))
    assert feature_pca_path.exists() and _is_non_trivial(feature_pca_path)

    # 2) depth
    depth_path = save_depth_colormap(depth[0], tmp_path / "depth_view0.png")
    assert depth_path.exists() and _is_non_trivial(depth_path)

    # 3) cross-attention heatmap
    ys = torch.arange(h).view(h, 1).expand(h, w)
    xs = torch.arange(w).view(1, w).expand(h, w)
    pos = torch.stack([ys, xs], dim=-1).reshape(-1, 2).unsqueeze(0)
    cross = Mamba3CrossAttention(dim_q=C, dim_kv=C, num_heads=6, state_dim=32, variant="B")
    rope = RoPE2D(base_frequency=100.0)
    with torch.no_grad():
        _y, attn = cross(feats[0:1], feats[1:2], q_pos=pos, kv_pos=pos, rope=rope, return_attn=True)
    attn_path = save_cross_attention_heatmap(
        attn, images[1], kv_grid_hw=(h, w), query_index=(h // 2) * w + w // 2, path=tmp_path / "cross_attention.png"
    )
    assert attn_path.exists() and _is_non_trivial(attn_path)

    # 4) instance-seg overlay on synthetic COCO-style samples
    samples = []
    for i in range(3):
        img = torch.rand(3, img_size, img_size)
        mask = torch.zeros(1, img_size, img_size, dtype=torch.bool)
        mask[0, img_size // 4 : 3 * img_size // 4, img_size // 4 : 3 * img_size // 4] = True
        samples.append(SimpleNamespace(image=img, masks=mask))

    head = TinyInstanceSegHead(in_channels=C, hidden=32, image_size=img_size)

    def extractor(img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            grid = net.features_grid(img.unsqueeze(0).unsqueeze(0))
        return grid[0]

    train_seg_head(head, extractor, samples, iters=20, lr=3e-3, device="cpu")
    seg_path = save_seg_overlay(head, extractor, samples[0], tmp_path / "seg_overlay_coco0.png")
    assert seg_path.exists() and _is_non_trivial(seg_path)
