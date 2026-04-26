"""Accuracy probe — does the SSD-based global-attention swap produce features
similar to DA3's transformer-based global attention?

PLAN § 15.45. Counterpart to scripts/bench_efficiency.py: that one measures
*efficiency* of the swap; this one measures *accuracy* of the swap, *before*
training.

Procedure
---------
1. Load DA3-SMALL (12-block transformer ViT-S with cross-view alternation
   from layer alt_start=4 onward — layers 5/7/9/11 are global cross-view).
2. Build our full-swap SSM-3D backbone (same shape, same alt_start,
   cat_token=True). All Mamba-3 self-attention modules.
3. Align weights:
     - load_da3_backbone() copies patch_embed, MLPs, norms, RoPE freqs
       from DA3 into our backbone.
     - warm_start_mamba3_from_qkv() initializes Mamba-3 B/C/V projections
       from DA3's qkv weights so the mixer starts in a state geometrically
       comparable to the transformer.
4. Feed identical multi-view input through both. Capture per-layer outputs.
5. Compare:
     - Per-layer cosine similarity (token-wise mean).
     - Per-layer effective_rank (both must produce rank-rich features).
     - Cross-view leakage: zero out view 0, re-run, measure how much
       view 1's tokens change at each cross-view layer. A working
       cross-view operation should leak some view-0 information into
       view-1 tokens; a broken one would show zero leakage.
6. Print a per-layer table; the cross-view layers (5/7/9/11) are the load-
   bearing rows for the global-attention-swap accuracy claim.

Note: this is at-init accuracy, not trained accuracy. The trained accuracy
answer is Step 4 (CM-FS-12 / FS-24 + eval on depth/F-score/pose-AUC).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssm3d.eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ssm3d.eval.metrics import effective_rank
from ssm3d.model import SSM3DBackbone
from ssm3d.weights import load_da3_backbone, warm_start_mamba3_from_qkv


def _hook_blocks(model, blocks_attr_path: list[str]):
    """Attach forward hooks on each block in `model.<blocks_attr_path>` and
    return (handles, captured dict keyed by layer index)."""
    obj = model
    for a in blocks_attr_path:
        obj = getattr(obj, a)
    captured: dict[int, torch.Tensor] = {}

    def hook(idx):
        def f(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[idx] = t.detach()
        return f

    handles = [b.register_forward_hook(hook(i)) for i, b in enumerate(obj)]
    return handles, captured


@torch.inference_mode()
def per_layer_outputs(model, x, blocks_attr_path):
    handles, captured = _hook_blocks(model, blocks_attr_path)
    try:
        if hasattr(model, "vit"):
            _ = model(x)               # SSM3DBackbone
        else:
            _ = model.model(x)         # DA3 wrapper
    finally:
        for h in handles:
            h.remove()
    return captured


def cosine_per_token_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    """a, b shape: (..., T, C) or (..., S, T, C). Compute cosine sim per
    token and average."""
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


def mean_effective_rank(t: torch.Tensor) -> float:
    """t shape: (..., T, C). Compute per-image effective_rank, mean."""
    if t.dim() == 4:                  # (B, S, T, C)
        flat = t.reshape(-1, t.shape[-2], t.shape[-1])
    elif t.dim() == 3:                # (N, T, C)
        flat = t
    else:                             # treat as one cloud
        flat = t.reshape(1, *t.shape[-2:])
    ranks = [effective_rank(flat[i]) for i in range(flat.shape[0])]
    return sum(ranks) / len(ranks) if ranks else float("nan")


def _slice_view(t: torch.Tensor, S: int, view_idx: int) -> torch.Tensor:
    """Pull a single view's tokens out of a captured block output.

    Local layers: shape (B*S, T, C). Global layers: shape (B, S*T+extras, C).
    Returns the per-view slice with shape (T, C).
    """
    if t.dim() == 3 and t.shape[0] == S:
        return t[view_idx]
    if t.dim() == 3:
        # Global: (B, S*T+extras, C). Strip cls + register prefix per DA3 layout
        # (1 cls token at index 0). Patches per view = (T_total - 1) // S.
        T_total = t.shape[1]
        per_view = (T_total - 1) // S
        start = 1 + view_idx * per_view
        return t[0, start:start + per_view]
    raise ValueError(f"unexpected tensor shape {t.shape}")


@torch.inference_mode()
def run_probe(args) -> None:
    device = args.device
    torch.manual_seed(args.seed)

    da3 = load_da3(DEFAULT_HF_MODEL, device=device)
    print(f"[1/4] DA3-SMALL loaded; backbone alt_start = "
          f"{da3.model.backbone.pretrained.alt_start}")

    ssm = SSM3DBackbone(
        size="small", img_size=args.img_size, patch_size=args.patch_size,
        depth=12, chunk_size=args.chunk_size, mamba_state_dim=args.state_dim,
        alt_start=4, cat_token=True,
    ).to(device).eval()
    print("[2/4] SSM-3D full-swap backbone built (alt_start=4, cat_token=True)")

    load_da3_backbone(ssm.vit, da3, verbose=False)
    da3_state = da3.model.backbone.pretrained.state_dict()
    warm_start_mamba3_from_qkv(
        ssm.vit, da3_state, prefix="",
        num_tokens=(args.img_size // args.patch_size) ** 2, verbose=False,
    )
    print("[3/4] weights aligned (DA3 non-attn loaded; Mamba-3 B/C/V warm-started from DA3 qkv)")

    n_views = args.n_views
    H = W = args.img_size
    x = torch.randn(1, n_views, 3, H, W, device=device)

    print(f"[4/4] feeding multi-view input (B=1, S={n_views}, {H}×{W})")
    da3_out = per_layer_outputs(da3, x, ["model", "backbone", "pretrained", "blocks"])
    ssm_out = per_layer_outputs(ssm, x, ["vit", "blocks"])

    # Cross-view layers in DA3 with alt_start=4: i ≥ 4 and i % 2 == 1 → 5, 7, 9, 11
    cross_view_layers = {5, 7, 9, 11}

    print(f"\n{'layer':>5} {'mode':<11} "
          f"{'cos_mean':>10} {'rank_da3':>10} {'rank_ssm':>10} {'rank_ratio':>11}")
    print("-" * 70)
    for i in range(12):
        mode = "cross-view" if i in cross_view_layers else "per-view"
        d, s = da3_out[i], ssm_out[i]
        # Rank: pull view-0 tokens out of each (handles different shapes for global layers)
        d_v0 = _slice_view(d, n_views, 0)
        s_v0 = _slice_view(s, n_views, 0)
        cos = cosine_per_token_mean(d_v0, s_v0)
        r_d = mean_effective_rank(d_v0)
        r_s = mean_effective_rank(s_v0)
        ratio = r_s / r_d if r_d > 0 else float("nan")
        marker = "  ←" if i in cross_view_layers else ""
        print(f"{i:>5} {mode:<11} {cos:>10.4f} {r_d:>10.2f} {r_s:>10.2f} {ratio:>11.3f}{marker}")

    # Cross-view information leakage probe: zero view 0, see how view-1
    # tokens change at each cross-view layer.
    print("\nCross-view information leakage probe")
    print("(zero view-0 input → measure layer-N view-1-token change)")
    print(f"{'layer':>5} {'mode':<11} {'da3 Δview-1':>12} {'ssm Δview-1':>12}")
    print("-" * 50)
    x_zeroed = x.clone()
    x_zeroed[:, 0] = 0.0
    da3_zero = per_layer_outputs(da3, x_zeroed, ["model", "backbone", "pretrained", "blocks"])
    ssm_zero = per_layer_outputs(ssm, x_zeroed, ["vit", "blocks"])
    for i in range(12):
        mode = "cross-view" if i in cross_view_layers else "per-view"
        d_v1 = _slice_view(da3_out[i], n_views, 1)
        d_v1_z = _slice_view(da3_zero[i], n_views, 1)
        s_v1 = _slice_view(ssm_out[i], n_views, 1)
        s_v1_z = _slice_view(ssm_zero[i], n_views, 1)
        d_diff = (d_v1 - d_v1_z).norm() / d_v1.norm().clamp_min(1e-8)
        s_diff = (s_v1 - s_v1_z).norm() / s_v1.norm().clamp_min(1e-8)
        marker = "  ←" if i in cross_view_layers else ""
        print(f"{i:>5} {mode:<11} {d_diff.item():>12.4f} {s_diff.item():>12.4f}{marker}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
