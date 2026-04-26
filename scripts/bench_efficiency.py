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

def build_attention_backbone(
    img_size: int, patch_size: int = 14, depth: int = 12,
    alt_start: int = -1, cat_token: bool = False,
) -> torch.nn.Module:
    """ViT-S with standard self-attention (default `Block.attn_class=Attention`).

    `alt_start=4, cat_token=True` mirrors DA3-SMALL's hybrid alternation; cross-view
    layers run self-attention on (B, S*N, C). `alt_start=-1` is per-view-only.
    """
    return vit_small(
        img_size=img_size,
        patch_size=patch_size,
        depth=depth,
        alt_start=alt_start,
        cat_token=cat_token,
        drop_path_rate=0.0,
    )


def build_ssd_backbone(
    img_size: int, patch_size: int = 14, depth: int = 12, chunk_size: int = 128,
    state_dim: int = 64, alt_start: int = -1, cat_token: bool = False,
) -> SSM3DBackbone:
    return SSM3DBackbone(
        size="small",
        img_size=img_size,
        patch_size=patch_size,
        depth=depth,
        chunk_size=chunk_size,
        mamba_state_dim=state_dim,
        alt_start=alt_start,
        cat_token=cat_token,
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


def _measure_variant(
    label: str, build_fn, x: torch.Tensor, T_patches: int, depth: int, state_dim: int,
    is_attn: bool, alt_start: int, device: torch.device, chunk_size: int,
) -> dict:
    """One backbone build + measure cycle."""
    img_size = int(x.shape[-1])
    try:
        net = build_fn().to(device).eval()
        n_params = count_params(net)
        with torch.inference_mode():
            if is_attn:
                fwd = lambda inp: net.get_intermediate_layers(inp, n=1, export_feat_layers=[])
            else:
                fwd = lambda inp: net(inp)
            metrics = measure(fwd, x, device)
        if is_attn:
            fl = attention_flops(T_patches, dim=384, num_heads=6, mlp_ratio=4.0, depth=depth)
        else:
            fl = ssd_flops(T_patches, dim=384, num_heads=6, state_dim=state_dim,
                           mlp_ratio=4.0, depth=depth, chunk_size=chunk_size)
        row = {
            "model": label, "img_size": img_size, "tokens": T_patches,
            "params_M": n_params / 1e6, "OOM": False, "peak_MiB": metrics["peak_mib"],
            "latency_ms": metrics["latency_ms"], "flops_G": fl["total"] / 1e9,
        }
        del net
        gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
        return row
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        return {"model": label, "img_size": img_size, "tokens": T_patches, "OOM": True}


def run_one(
    img_size: int, patch_size: int, depth: int,
    chunk_size: int, state_dim: int, device: torch.device,
    n_views: int = 12,
) -> dict:
    """Build all four backbones at this img_size, measure, return rows for the table.

    Variants:
    - SSD partial-swap (alt_start=-1): legacy CM12-CM30 backbone, per-view only.
    - SSD full-swap (alt_start=4, cat_token=True): DA3-faithful with cross-view.
    - Attention partial-swap: per-view only.
    - Attention full-swap (DA3-SMALL native): the apples-to-apples baseline.
    """
    rows: list[dict] = []
    H = W = img_size
    T_patches = (img_size // patch_size) ** 2
    # Multi-view input (B=1, S=n_views) so cross-view layers actually run.
    x = torch.randn(1, n_views, 3, H, W, device=device)

    rows.append(_measure_variant(
        "SSD partial-swap (per-view)",
        lambda: build_ssd_backbone(
            img_size=img_size, patch_size=patch_size, depth=depth,
            chunk_size=chunk_size, state_dim=state_dim,
            alt_start=-1, cat_token=False,
        ),
        x, T_patches, depth, state_dim, is_attn=False, alt_start=-1,
        device=device, chunk_size=chunk_size,
    ))

    rows.append(_measure_variant(
        "SSD full-swap (alt_start=4)",
        lambda: build_ssd_backbone(
            img_size=img_size, patch_size=patch_size, depth=depth,
            chunk_size=chunk_size, state_dim=state_dim,
            alt_start=4, cat_token=True,
        ),
        x, T_patches, depth, state_dim, is_attn=False, alt_start=4,
        device=device, chunk_size=chunk_size,
    ))

    rows.append(_measure_variant(
        "Self-attn partial-swap",
        lambda: build_attention_backbone(
            img_size=img_size, patch_size=patch_size, depth=depth,
            alt_start=-1, cat_token=False,
        ),
        x, T_patches, depth, state_dim, is_attn=True, alt_start=-1,
        device=device, chunk_size=chunk_size,
    ))

    rows.append(_measure_variant(
        "Self-attn full-swap (DA3 native)",
        lambda: build_attention_backbone(
            img_size=img_size, patch_size=patch_size, depth=depth,
            alt_start=4, cat_token=True,
        ),
        x, T_patches, depth, state_dim, is_attn=True, alt_start=4,
        device=device, chunk_size=chunk_size,
    ))

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 392, 504, 1022])
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--n-views", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"\nbackbone-only forward, B=1 batch × S={args.n_views} views, depth={args.depth}, "
          f"patch={args.patch_size}, chunk={args.chunk_size}, state_dim={args.state_dim}\n")
    print(f"{'model':<36} {'img':>5} {'tokens':>7} {'params(M)':>10} "
          f"{'peak(MiB)':>11} {'latency(ms)':>13} {'FLOPs(G)':>10}")
    print("-" * 100)

    all_rows: list[dict] = []
    for s in args.sizes:
        rows = run_one(
            img_size=s, patch_size=args.patch_size, depth=args.depth,
            chunk_size=args.chunk_size, state_dim=args.state_dim, device=device,
            n_views=args.n_views,
        )
        for r in rows:
            if r.get("OOM"):
                print(f"{r['model']:<36} {r['img_size']:>5} {r['tokens']:>7} "
                      f"{'—':>10} {'OOM':>11} {'—':>13} {'—':>10}")
            else:
                print(f"{r['model']:<36} {r['img_size']:>5} {r['tokens']:>7} "
                      f"{r['params_M']:>10.2f} {r['peak_MiB']:>11.1f} "
                      f"{r['latency_ms']:>13.2f} {r['flops_G']:>10.2f}")
            all_rows.append(r)

    # Apples-to-apples: full-swap SSD vs full-swap self-attn (DA3 native).
    print("\nFull-swap SSD / Self-attn (DA3 native) ratios — load-bearing comparison:")
    print(f"{'img':>5} {'tokens':>7} {'mem ratio':>11} {'lat ratio':>11} {'FLOPs ratio':>12}")
    print("-" * 50)
    by_size: dict[int, dict] = {}
    for r in all_rows:
        by_size.setdefault(r["img_size"], {})[r["model"]] = r
    for s in args.sizes:
        ssd = by_size[s].get("SSD full-swap (alt_start=4)", {})
        attn = by_size[s].get("Self-attn full-swap (DA3 native)", {})
        if ssd.get("OOM") or attn.get("OOM") or not ssd or not attn:
            label_oom = "OOM" if (ssd.get("OOM") or attn.get("OOM")) else ""
            print(f"{s:>5} {ssd.get('tokens', '—'):>7} {label_oom:>11} {label_oom:>11} "
                  f"{label_oom:>12}")
            continue
        mem_ratio = ssd["peak_MiB"] / attn["peak_MiB"]
        lat_ratio = ssd["latency_ms"] / attn["latency_ms"]
        flops_ratio = ssd["flops_G"] / attn["flops_G"]
        print(f"{s:>5} {ssd['tokens']:>7} {mem_ratio:>11.2f}× {lat_ratio:>11.2f}× "
              f"{flops_ratio:>12.2f}×")


if __name__ == "__main__":
    main()
