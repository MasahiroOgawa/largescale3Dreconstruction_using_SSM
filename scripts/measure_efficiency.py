#!/usr/bin/env python3
"""Parameters, peak GPU memory and latency for DA3-SMALL against each swapped mixer.

This is the half of the ETH3D comparison that accuracy metrics cannot answer: whether
Vision Mamba-3 is cheaper than the softmax attention it replaces, at matched parameters.
None of it depends on trained weights -- memory and latency are properties of the
architecture -- so it runs on freshly constructed models in minutes and needs no
checkpoint, no distillation and no TSDF fusion.

Swept over image size because that is the axis where the claim has content: the operators
differ as O(T) against O(T^2) in token count, and at 504/14 a single resolution shows only
one point on that curve. A softmax arm that OOMs at the largest size is a result, reported
as such rather than hidden.

  uv run python scripts/measure_efficiency.py --sizes 252 504 756
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mamba3_attn.eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from mamba3_attn.patch import install_mamba3

# (label, install_mamba3 variant or None for un-patched DA3-SMALL, rope_all_layers)
# The baseline is un-patched and keeps DA3's own RoPE layout, exactly as published; the
# swapped arms carry 2-D RoPE on every backbone block, which is part of the operator's
# design (VSSD's mask has no |i-j| term, so it needs one to function).
ARMS = [
    ("DA3-SMALL (softmax)", None, False),
    ("bidirectional SSD", "mamba3", True),
    ("VSSD-gamma", "vssd", True),
    ("VSSD-beta,gamma", "vssd_bg", True),
]


def build(variant: str | None, rope_all: bool, device: str):
    api = load_da3(DEFAULT_HF_MODEL, device=device)
    if variant is not None:
        install_mamba3(api.model, which="all", variant=variant, state_dim=64,
                       use_fused_kernel=False, rope_all_layers=rope_all)
    return api.model.to(device).eval()


@torch.no_grad()
def measure(model, size: int, device: str, warmup: int = 3, iters: int = 10) -> dict:
    """Peak allocated memory and median latency for one forward pass.

    Median rather than mean: a single slow iteration from a driver hiccup or a background
    process should not move the number. Memory is read from max_memory_allocated after a
    reset, so it is this model's own peak and not whatever the previous arm left behind.
    """
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warmup):
        model(x.unsqueeze(0))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x.unsqueeze(0))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return {
        "latency_ms": times[len(times) // 2] * 1e3,
        "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[252, 504, 756])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=Path, default=Path("result/efficiency.json"))
    args = ap.parse_args()

    rows = []
    for label, variant, rope_all in ARMS:
        model = build(variant, rope_all, args.device)
        n_params = sum(p.numel() for p in model.parameters())
        for size in args.sizes:
            tokens = (size // 14) ** 2
            try:
                m = measure(model, size, args.device)
                print(f"{label:22s} {size:4d}px  T={tokens:5d}  "
                      f"{m['latency_ms']:8.1f} ms  {m['peak_mib']:8.1f} MiB", flush=True)
                rows.append({"arm": label, "size": size, "tokens": tokens,
                             "params_m": n_params / 1e6, **m})
            except torch.OutOfMemoryError:
                # A softmax arm that cannot fit the largest size is the point of the
                # sweep, so it is recorded rather than allowed to abort the run.
                torch.cuda.empty_cache()
                print(f"{label:22s} {size:4d}px  T={tokens:5d}  OOM", flush=True)
                rows.append({"arm": label, "size": size, "tokens": tokens,
                             "params_m": n_params / 1e6, "oom": True})
        del model
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
