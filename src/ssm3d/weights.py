"""Load DINOv2 non-attention pretrained weights into the SSM3D backbone.

The Mamba-3 attention blocks have different parameters (B, C, V, Δ, A, λ
projections) than DINOv2's qkv attention, so the `.attn.` keys are always
filtered out. Everything else (patch embed, LayerNorms, MLPs, cls/register
tokens, 2D-RoPE freqs) matches DA3's DINOv2 ViT layout and loads cleanly
whenever the shapes agree.

Shape mismatches (e.g. patch_embed for patch_size=16 vs 14 DINOv2 default)
are filtered silently; a summary is printed so the caller can see how many
keys loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import nn


def _read_state_dict(source: Union[str, Path, dict]) -> dict:
    if isinstance(source, dict):
        return source
    path = Path(source)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    return obj


def load_dinov2_backbone(
    vit: nn.Module,
    source: Union[str, Path, dict],
    prefix: str = "",
    verbose: bool = True,
) -> dict[str, int]:
    """Load DINOv2 weights into `vit`, filtering .attn.* and shape mismatches.

    Args:
        vit: the `DinoVisionTransformer` instance (e.g. `SSM3DBackbone.vit`).
        source: state dict, or path to a checkpoint file.
        prefix: prefix to strip from checkpoint keys before loading
            (some checkpoints have "backbone." or "pretrained.").
        verbose: print a one-line summary.

    Returns:
        dict with counters: {"loaded", "skipped_attn", "shape_mismatch",
            "missing_in_ckpt", "total_target"}
    """
    src = _read_state_dict(source)
    if prefix:
        src = {k[len(prefix):]: v for k, v in src.items() if k.startswith(prefix)}

    target = vit.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped_attn = 0
    shape_mismatch = 0
    for k, v in src.items():
        if ".attn." in k:
            skipped_attn += 1
            continue
        if k not in target:
            continue
        if target[k].shape != v.shape:
            shape_mismatch += 1
            continue
        loaded[k] = v

    vit.load_state_dict(loaded, strict=False)

    counters = {
        "loaded": len(loaded),
        "skipped_attn": skipped_attn,
        "shape_mismatch": shape_mismatch,
        "missing_in_ckpt": sum(1 for k in target if k not in src and ".attn." not in k),
        "total_target": len(target),
    }
    if verbose:
        print(
            f"[weights] loaded {counters['loaded']}/{counters['total_target']} "
            f"tensors  (skipped_attn={counters['skipped_attn']}, "
            f"shape_mismatch={counters['shape_mismatch']}, "
            f"missing_in_ckpt={counters['missing_in_ckpt']})"
        )
    return counters
