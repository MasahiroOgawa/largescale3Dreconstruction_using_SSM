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

import math
from pathlib import Path
from typing import Optional, Union

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


def load_da3_backbone(
    vit: nn.Module,
    da3_model,
    verbose: bool = True,
) -> dict[str, int]:
    """Load DA3's published backbone weights (non-attention) into our ViT.

    DA3 fine-tuned its DINOv2 backbone jointly with the DualDPT head. Loading
    PatchEmbed, MLPs, norms, and RoPE freqs from the DA3 checkpoint (instead of
    upstream DINOv2) gives the SSM-3D backbone a head start that is one
    attention-training-step away from the teacher, which is what Phase B will
    close (PLAN §9, R6).

    Uses the same .attn.* filter as `load_dinov2_backbone` — the Mamba-3 mixer
    has no DA3-compatible attention params to load.

    DA3 stores the inner DINOv2 under `self.pretrained`, so keys arrive as
    `pretrained.blocks.*`, `pretrained.patch_embed.*`, etc. We strip that
    prefix before the load so keys align with our bare `DinoVisionTransformer`.
    """
    backbone_state = da3_model.model.backbone.state_dict()
    return load_dinov2_backbone(vit, backbone_state, prefix="pretrained.", verbose=verbose)


def _uniform_delta_bias(num_tokens: int) -> float:
    """Pre-softplus value so softplus(bias) ≈ 1/T.

    The SSD mask has sum_j L[t,j] ≈ Σ_j Δ_j when α ≈ 1. For ~uniform day-0
    attention over T tokens we want Δ ≈ 1/T, hence softplus(b) = 1/T and
    b ≈ log(exp(1/T) − 1). For T ≳ 20 this is well-approximated by log(1/T).
    """
    target = 1.0 / max(num_tokens, 1)
    return math.log(math.expm1(target))


@torch.no_grad()
def warm_start_mamba3_from_qkv(
    vit: nn.Module,
    source: Union[str, Path, dict],
    prefix: str = "",
    verbose: bool = True,
    num_tokens: Optional[int] = None,
) -> dict[str, int]:
    """Cast DINOv2 softmax-attention Q/K/V into Mamba-3 C/B/V projections.

    For each block i, DINOv2 stores attention as `attn.qkv.weight` ∈ R^{3D×D}
    and `attn.proj.weight` ∈ R^{D×D}. Our Mamba-3 attention stores one big
    projection `attn.inner.projections.proj.weight` of shape (out_size, D)
    with row layout `[B | C | V | Δ | A | λ]` and an output linear
    `attn.inner.proj.weight` of shape (D, D).

    This function copies
      - K-rows → B-rows (per head, first `state_dim` of `head_dim`)
      - Q-rows → C-rows (per head, first `state_dim` of `head_dim`)
      - V-rows → V-rows (full `D`)
      - DINOv2 `attn.proj` → Mamba-3 `attn.inner.proj`
    It also sets `delta_bias` so that softplus(Δ_raw + bias) ≈ 1/num_tokens
    at day 0, producing an approximately uniform (soft-max-like) decay mask.
    `A_bias` and `lam_bias` stay at zero.

    Note (see PLAN §11): this warm-start empirically *worsens* feature cosine
    dispersion (0.14 → 0.97 on ETH3D) when the Mamba-3 attention is not
    subsequently trained. SSD attention lacks softmax's row normalisation,
    so structurally-correct-direction-but-wrong-scale outputs corrupt
    DINOv2's MLP activation distribution worse than small-magnitude noise.
    Intended for use as a starting point for downstream training, not as a
    drop-in replacement for the pretrained attention at inference time.

    Blocks whose `state_dim > head_dim` or whose ckpt qkv is missing are skipped.

    Args:
        num_tokens: sequence length used to size the uniform-attention Δ bias.
            Defaults to `vit.patch_embed.num_patches` if available, else 256.
    """
    src = _read_state_dict(source)
    if prefix:
        src = {k[len(prefix):]: v for k, v in src.items() if k.startswith(prefix)}

    warmed = 0
    skipped = 0
    out_warmed = 0

    blocks = getattr(vit, "blocks", None)
    if blocks is None:
        if verbose:
            print("[warm_start] vit has no `blocks`; nothing to do")
        return {"warmed": 0, "skipped": 0, "out_warmed": 0, "blocks": 0}

    if num_tokens is None:
        # Default to the patch grid size if available, else fall back to 256.
        if hasattr(vit, "patch_embed") and hasattr(vit.patch_embed, "num_patches"):
            num_tokens = int(vit.patch_embed.num_patches)
        else:
            num_tokens = 256
    delta_bias_val = _uniform_delta_bias(num_tokens)

    for block_idx, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        inner = getattr(attn, "inner", None) if attn is not None else None
        projections = getattr(inner, "projections", None) if inner is not None else None
        if projections is None:
            skipped += 1
            continue

        qkv_key = f"blocks.{block_idx}.attn.qkv.weight"
        if qkv_key not in src:
            skipped += 1
            continue
        qkv = src[qkv_key]
        D = projections.dim
        H = projections.num_heads
        N = projections.state_dim
        hd = projections.head_dim

        if qkv.shape != (3 * D, D) or hd < N:
            skipped += 1
            continue

        Q = qkv[0:D].view(H, hd, D)[:, :N, :].reshape(H * N, D)
        K = qkv[D : 2 * D].view(H, hd, D)[:, :N, :].reshape(H * N, D)
        V = qkv[2 * D : 3 * D]

        target = projections.proj.weight.data
        s1 = H * N
        s2 = H * N
        target[:s1].copy_(K)
        target[s1 : s1 + s2].copy_(Q)
        target[s1 + s2 : s1 + s2 + D].copy_(V)

        # Day-0 uniform attention: softplus(delta_raw + delta_bias) ≈ 1/T, so
        # the mask L[t,j] is near-constant along j (no strong recency bias).
        projections.delta_bias.data.fill_(delta_bias_val)
        # A_bias and lam_bias stay at zero (softplus(0)·(−1) ≈ −0.69 and
        # sigmoid(0) = 0.5 — mild α, balanced trapezoid).
        warmed += 1

        out_w = src.get(f"blocks.{block_idx}.attn.proj.weight")
        if out_w is not None and inner.proj.weight.shape == out_w.shape:
            inner.proj.weight.data.copy_(out_w)
            out_b = src.get(f"blocks.{block_idx}.attn.proj.bias")
            if (
                out_b is not None
                and inner.proj.bias is not None
                and inner.proj.bias.shape == out_b.shape
            ):
                inner.proj.bias.data.copy_(out_b)
            out_warmed += 1

    counters = {
        "warmed": warmed,
        "skipped": skipped,
        "out_warmed": out_warmed,
        "blocks": len(blocks),
    }
    if verbose:
        print(
            f"[warm_start] warmed {warmed}/{counters['blocks']} blocks  "
            f"(out_proj warmed={out_warmed}, skipped={skipped})"
        )
    return counters
