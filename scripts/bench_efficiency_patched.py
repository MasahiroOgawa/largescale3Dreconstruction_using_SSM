"""Efficiency benchmark — patched DA3 (Mamba-3) vs un-patched DA3 (transformer).

Apples-to-apples: same DA3-SMALL model, only difference = attention modules.
This is the load-bearing efficiency comparison for the paper's claim:
"Mamba-3 SSD attention as a drop-in replacement for transformer attention
in DA3, achieving X% of DA3-SMALL quality at Y× lower memory / J× faster
inference at K× longer T."

Measures the **full DA3 model forward** (backbone + DPT head + cam_dec)
on multi-view input (B=1, S=n_views), not just the backbone — matches
the actual deployment cost.

OOM is treated as a data point: at large T, transformer attention OOMs
where Mamba-3 (linear-in-T) succeeds.

Example:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run python scripts/bench_efficiency_patched.py \\
            --sizes 224 392 504 --n-views 4 8 12
"""
from __future__ import annotations

import argparse
import gc
import time
from contextlib import contextmanager

import torch

from mamba3_attn.eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from mamba3_attn.patch import install_mamba3


@contextmanager
def cuda_mem_scope(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(forward_fn, x: torch.Tensor, device: torch.device,
            warmup: int = 2, repeats: int = 5) -> dict:
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
    median = timings_ms[len(timings_ms) // 2]
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda" else float("nan")
    )
    return {"latency_ms": median, "peak_mib": peak}


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def measure_variant(label: str, build_fn, x: torch.Tensor, device: torch.device) -> dict:
    img_size = int(x.shape[-1])
    n_views = int(x.shape[1])
    try:
        model = build_fn().to(device).eval()
        n_params = count_params(model.model)
        with torch.inference_mode():
            metrics = measure(lambda inp: model.model(inp), x, device)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "label": label, "img_size": img_size, "n_views": n_views,
            "params_M": n_params / 1e6, "OOM": False,
            "peak_MiB": metrics["peak_mib"], "latency_ms": metrics["latency_ms"],
        }
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"label": label, "img_size": img_size, "n_views": n_views, "OOM": True}


def build_da3_unpatched(device: str):
    return load_da3(DEFAULT_HF_MODEL, device=device)


def build_da3_patched(device: str, state_dim: int, use_fused_kernel: bool):
    api = load_da3(DEFAULT_HF_MODEL, device=device)
    install_mamba3(
        api.model, which="all", state_dim=state_dim,
        use_fused_kernel=use_fused_kernel, chunk_size=128,
    )
    return api.to(device)


def run_grid(sizes: list[int], n_views_list: list[int], state_dim: int,
             device: torch.device) -> list[dict]:
    rows: list[dict] = []
    for img_size in sizes:
        for n_views in n_views_list:
            x = torch.randn(1, n_views, 3, img_size, img_size, device=device)
            tokens_per_view = (img_size // 14) ** 2
            T_total = n_views * tokens_per_view
            print(f"\n[bench] img={img_size}² S={n_views}  tokens/view={tokens_per_view}  cross-view T={T_total}")

            r1 = measure_variant(
                "DA3-SMALL (transformer)",
                lambda: build_da3_unpatched(device.type),
                x, device,
            )
            r1["tokens_per_view"] = tokens_per_view
            r1["cross_view_T"] = T_total
            rows.append(r1)
            _print_row(r1)

            r2 = measure_variant(
                "DA3-SMALL +Mamba-3 (PyTorch)",
                lambda: build_da3_patched(device.type, state_dim, use_fused_kernel=False),
                x, device,
            )
            r2["tokens_per_view"] = tokens_per_view
            r2["cross_view_T"] = T_total
            rows.append(r2)
            _print_row(r2)

            r3 = measure_variant(
                "DA3-SMALL +Mamba-3 (Triton kernel)",
                lambda: build_da3_patched(device.type, state_dim, use_fused_kernel=True),
                x, device,
            )
            r3["tokens_per_view"] = tokens_per_view
            r3["cross_view_T"] = T_total
            rows.append(r3)
            _print_row(r3)

            del x
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def _print_row(r: dict) -> None:
    label = r["label"]
    img = r["img_size"]; nv = r["n_views"]
    if r.get("OOM"):
        print(f"  {label:<40} img={img}² S={nv}  OOM")
        return
    print(f"  {label:<40} img={img}² S={nv}  "
          f"params={r['params_M']:.2f}M  peak={r['peak_MiB']:.1f}MiB  "
          f"lat={r['latency_ms']:.1f}ms")


def summary_ratios(rows: list[dict], sizes: list[int], n_views_list: list[int]) -> None:
    by_key: dict[tuple[int, int, str], dict] = {}
    for r in rows:
        by_key[(r["img_size"], r["n_views"], r["label"])] = r

    print("\n" + "=" * 80)
    print("Apples-to-apples: DA3-SMALL +Mamba-3 (Triton) vs DA3-SMALL (transformer)")
    print("=" * 80)
    print(f"{'img':>5} {'S':>3} {'cross-T':>8} {'mem ratio':>10} {'lat ratio':>10}")
    print("-" * 45)
    for img in sizes:
        for nv in n_views_list:
            attn = by_key.get((img, nv, "DA3-SMALL (transformer)"))
            ssd = by_key.get((img, nv, "DA3-SMALL +Mamba-3 (Triton kernel)"))
            if not attn or not ssd:
                continue
            T = nv * (img // 14) ** 2
            if attn.get("OOM") and ssd.get("OOM"):
                tag = "both OOM"
            elif attn.get("OOM"):
                tag = f"attn OOM, mamba ok ({ssd['peak_MiB']:.0f}MiB / {ssd['latency_ms']:.0f}ms)"
            elif ssd.get("OOM"):
                tag = "mamba OOM"
            else:
                mem_r = ssd["peak_MiB"] / attn["peak_MiB"]
                lat_r = ssd["latency_ms"] / attn["latency_ms"]
                tag = f"{mem_r:>10.2f}× {lat_r:>10.2f}×"
            print(f"{img:>5} {nv:>3} {T:>8}  {tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 392, 504])
    ap.add_argument("--n-views", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"\n[bench_efficiency_patched] DA3 full-model forward, B=1\n")

    rows = run_grid(args.sizes, args.n_views, args.state_dim, device)
    summary_ratios(rows, args.sizes, args.n_views)


if __name__ == "__main__":
    main()
