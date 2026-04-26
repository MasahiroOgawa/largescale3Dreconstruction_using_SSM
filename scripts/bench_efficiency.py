"""Efficiency benchmark — Mamba-3 SSD attention vs standard self-attention.

PLAN § 15.41 (eval-expansion Step A). Backbone-only comparison (cleanest
single-variable attention swap), measured at multiple input resolutions.

For each input size in `--sizes`:
  1. Build a 12-block ViT-S backbone with **default self-attention**.
  2. Build the same backbone with **Mamba-3 SSD attention** (our SSM3DBackbone).
  3. Forward-pass random input; record peak GPU memory + wall-clock latency.
  4. Compute theoretical FLOPs from architectural parameters.

Both backbones are 22 M params (same shape). Only the mixer differs.

OOM is treated as a *data point* for the transformer at large inputs:
the script catches the exception and reports "OOM".

Example:
    uv run python scripts/bench_efficiency.py --sizes 224 384 504 1024
"""
from __future__ import annotations

import argparse
import gc
import time
from contextlib import contextmanager

import torch

from ssm3d.model import SSM3DBackbone

# Standard DINOv2-S backbone (same shape, vanilla self-attention).
from depth_anything_3.model.dinov2.vision_transformer import vit_small


# ---------- model construction ----------

def build_attention_backbone(img_size: int, patch_size: int = 14, depth: int = 12) -> torch.nn.Module:
    """ViT-S with standard self-attention (default `Block.attn_class=Attention`)."""
    return vit_small(
        img_size=img_size,
        patch_size=patch_size,
        depth=depth,
        cat_token=False,
        drop_path_rate=0.0,
    )


def build_ssd_backbone(
    img_size: int, patch_size: int = 14, depth: int = 12, chunk_size: int = 128,
    state_dim: int = 64,
) -> SSM3DBackbone:
    return SSM3DBackbone(
        size="small",
        img_size=img_size,
        patch_size=patch_size,
        depth=depth,
        chunk_size=chunk_size,
        mamba_state_dim=state_dim,
    )


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------- FLOP counting (manual, asymptotic) ----------

def attention_flops(
    T: int, dim: int, num_heads: int, mlp_ratio: float, depth: int,
) -> dict:
    """FLOPs per forward pass, per block, summed across `depth` blocks.

    Self-attention block: 4·T·D² (qkv + proj) + 2·T²·D (scores + apply) + MLP
    MLP (default 4× expand): 2 · T · D · (mlp_ratio·D) + 2 · T · (mlp_ratio·D) · D
                           = 4 · mlp_ratio · T · D²
    """
    D = dim
    qkv_proj = 3 * T * D * D
    out_proj = T * D * D
    scores = T * T * D            # Q @ K^T
    apply = T * T * D             # softmax · V
    mlp = 2 * mlp_ratio * T * D * D + 2 * mlp_ratio * T * D * D
    per_block = qkv_proj + out_proj + scores + apply + mlp
    return {
        "per_block": per_block,
        "total": per_block * depth,
        "attn_quadratic_term": (scores + apply) * depth,    # the O(T^2) part
        "linear_term": (qkv_proj + out_proj + mlp) * depth, # the O(T) part
    }


def ssd_flops(
    T: int, dim: int, num_heads: int, state_dim: int, mlp_ratio: float, depth: int,
    chunk_size: int = 128,
) -> dict:
    """FLOPs for SSD attention block (linear in T).

    Per layer:
      B/C/V projections: 3 · T · D · D = 3·T·D²
      A/lam/delta scalars: T · D (small, ignored)
      output projection: T · D · D = T·D²
      SSD scan (chunked, length T, state_dim N, num_heads H):
        per chunk: c² · H · N (mask) + c · H · N · D (apply) ≈ c · D · H · N (linearly summed)
      sum over T/chunk_size chunks: T · D · H · N + T · chunk_size · H · N
      MLP (default 4× expand): 4 · mlp_ratio · T · D²
    """
    D = dim
    H = num_heads
    N = state_dim
    bcv_proj = 3 * T * D * D
    out_proj = T * D * D
    ssd_apply = T * D * H * N
    ssd_mask = (T // chunk_size + 1) * chunk_size * chunk_size * H * N
    mlp = 4 * mlp_ratio * T * D * D
    per_block = bcv_proj + out_proj + ssd_apply + ssd_mask + mlp
    return {
        "per_block": per_block,
        "total": per_block * depth,
        "ssd_term": (ssd_apply + ssd_mask) * depth,        # the O(T) + O(T·c) part
        "linear_term": (bcv_proj + out_proj + mlp) * depth,
    }


# ---------- benchmarking ----------

@contextmanager
def cuda_mem_scope(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    forward_fn, x: torch.Tensor, device: torch.device,
    warmup: int = 3, repeats: int = 20,
) -> dict:
    """Run forward_fn(x), report peak memory (MiB) and median latency (ms)."""
    for _ in range(warmup):
        forward_fn(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with cuda_mem_scope(device):
        timings_ms: list[float] = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            forward_fn(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
    timings_ms.sort()
    median_ms = timings_ms[len(timings_ms) // 2]
    peak_mib = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda" else float("nan")
    )
    return {"latency_ms": median_ms, "peak_mib": peak_mib}


def run_one(
    img_size: int, patch_size: int, depth: int,
    chunk_size: int, state_dim: int, device: torch.device,
) -> dict:
    """Build both backbones at this img_size, measure, return rows for the table."""
    rows: list[dict] = []

    # Random input. (B=1, S=1, 3, H, W) for the per-view backbone signature SSM3D uses.
    H = W = img_size
    T_patches = (img_size // patch_size) ** 2
    x = torch.randn(1, 1, 3, H, W, device=device)

    # SSD backbone
    try:
        ssd = build_ssd_backbone(
            img_size=img_size, patch_size=patch_size, depth=depth,
            chunk_size=chunk_size, state_dim=state_dim,
        ).to(device).eval()
        n_params_ssd = count_params(ssd)
        with torch.inference_mode():
            metrics = measure(lambda inp: ssd(inp), x, device)
        ssd_fl = ssd_flops(T_patches, dim=384, num_heads=6, state_dim=state_dim,
                           mlp_ratio=4.0, depth=depth, chunk_size=chunk_size)
        rows.append({
            "model": "SSD (Mamba-3)", "img_size": img_size, "tokens": T_patches,
            "params_M": n_params_ssd / 1e6, "OOM": False,
            "peak_MiB": metrics["peak_mib"], "latency_ms": metrics["latency_ms"],
            "flops_G": ssd_fl["total"] / 1e9,
        })
        del ssd
        gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    except torch.cuda.OutOfMemoryError:
        rows.append({"model": "SSD (Mamba-3)", "img_size": img_size, "tokens": T_patches,
                     "OOM": True})
        gc.collect(); torch.cuda.empty_cache()

    # Self-attention backbone (vanilla DINOv2-S layers)
    try:
        attn = build_attention_backbone(img_size=img_size, patch_size=patch_size, depth=depth)
        attn = attn.to(device).eval()
        n_params_attn = count_params(attn)
        with torch.inference_mode():
            metrics = measure(
                lambda inp: attn.get_intermediate_layers(inp, n=1, export_feat_layers=[]),
                x, device,
            )
        attn_fl = attention_flops(T_patches, dim=384, num_heads=6, mlp_ratio=4.0, depth=depth)
        rows.append({
            "model": "Self-attention (DINOv2-S)", "img_size": img_size, "tokens": T_patches,
            "params_M": n_params_attn / 1e6, "OOM": False,
            "peak_MiB": metrics["peak_mib"], "latency_ms": metrics["latency_ms"],
            "flops_G": attn_fl["total"] / 1e9,
        })
        del attn
        gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    except torch.cuda.OutOfMemoryError:
        rows.append({"model": "Self-attention (DINOv2-S)", "img_size": img_size,
                     "tokens": T_patches, "OOM": True})
        gc.collect(); torch.cuda.empty_cache()

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 384, 504, 1024])
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"\nbackbone-only forward pass, B=1 view, depth={args.depth}, "
          f"patch={args.patch_size}, chunk={args.chunk_size}, state_dim={args.state_dim}\n")
    print(f"{'model':<26} {'img':>5} {'tokens':>7} {'params(M)':>10} "
          f"{'peak(MiB)':>11} {'latency(ms)':>13} {'FLOPs(G)':>10}")
    print("-" * 90)

    all_rows: list[dict] = []
    for s in args.sizes:
        rows = run_one(
            img_size=s, patch_size=args.patch_size, depth=args.depth,
            chunk_size=args.chunk_size, state_dim=args.state_dim, device=device,
        )
        for r in rows:
            if r.get("OOM"):
                print(f"{r['model']:<26} {r['img_size']:>5} {r['tokens']:>7} "
                      f"{'—':>10} {'OOM':>11} {'—':>13} {'—':>10}")
            else:
                print(f"{r['model']:<26} {r['img_size']:>5} {r['tokens']:>7} "
                      f"{r['params_M']:>10.2f} {r['peak_MiB']:>11.1f} "
                      f"{r['latency_ms']:>13.2f} {r['flops_G']:>10.2f}")
            all_rows.append(r)

    # Side-by-side ratio summary
    print("\nSSD / Self-attention ratios (lower = SSD better):")
    print(f"{'img':>5} {'tokens':>7} {'mem ratio':>11} {'lat ratio':>11} {'FLOPs ratio':>12}")
    print("-" * 50)
    by_size: dict[int, dict] = {}
    for r in all_rows:
        by_size.setdefault(r["img_size"], {})[r["model"]] = r
    for s in args.sizes:
        ssd = by_size[s].get("SSD (Mamba-3)", {})
        attn = by_size[s].get("Self-attention (DINOv2-S)", {})
        if ssd.get("OOM") or attn.get("OOM") or not ssd or not attn:
            label_attn_oom = "OOM" if attn.get("OOM") else ""
            print(f"{s:>5} {ssd.get('tokens', '—'):>7} {label_attn_oom:>11} {label_attn_oom:>11} "
                  f"{label_attn_oom:>12}")
            continue
        mem_ratio = ssd["peak_MiB"] / attn["peak_MiB"]
        lat_ratio = ssd["latency_ms"] / attn["latency_ms"]
        flops_ratio = ssd["flops_G"] / attn["flops_G"]
        print(f"{s:>5} {ssd['tokens']:>7} {mem_ratio:>11.2f}× {lat_ratio:>11.2f}× "
              f"{flops_ratio:>12.2f}×")


if __name__ == "__main__":
    main()
