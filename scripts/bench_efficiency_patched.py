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
            --sizes 224 392 504 --n-views 4 8 12 \\
            --out-dir outputs/runs/scene_overfit_terrains_orig
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from contextlib import contextmanager
from pathlib import Path

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


def write_outputs(rows: list[dict], sizes: list[int], n_views_list: list[int],
                  out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "efficiency.json").write_text(json.dumps(rows, indent=2))

    by_key: dict[tuple[int, int, str], dict] = {
        (r["img_size"], r["n_views"], r["label"]): r for r in rows
    }
    md = ["# Efficiency comparison — DA3-SMALL transformer vs +Mamba-3\n",
          "\nFull-model forward (backbone + DPT head + cam_dec), B=1, GPU.\n",
          "Apples-to-apples: same architecture, only attention modules differ.\n",
          "\n| img | S | cross-T | DA3 transformer | +Mamba-3 (PyTorch) | +Mamba-3 (Triton) | mem ratio | lat ratio |\n",
          "|---|---|---|---|---|---|---|---|\n"]
    for img in sizes:
        for nv in n_views_list:
            T = nv * (img // 14) ** 2
            attn = by_key.get((img, nv, "DA3-SMALL (transformer)"))
            ssd_pt = by_key.get((img, nv, "DA3-SMALL +Mamba-3 (PyTorch)"))
            ssd_tr = by_key.get((img, nv, "DA3-SMALL +Mamba-3 (Triton kernel)"))
            def cell(r: dict | None) -> str:
                if r is None:
                    return "-"
                if r.get("OOM"):
                    return "**OOM**"
                return f"{r['peak_MiB']:.0f} MiB / {r['latency_ms']:.0f} ms"
            mem_r_str = lat_r_str = "-"
            if attn and ssd_tr and not attn.get("OOM") and not ssd_tr.get("OOM"):
                mem_r_str = f"{ssd_tr['peak_MiB'] / attn['peak_MiB']:.2f}×"
                lat_r_str = f"{ssd_tr['latency_ms'] / attn['latency_ms']:.2f}×"
            elif attn and attn.get("OOM") and ssd_tr and not ssd_tr.get("OOM"):
                mem_r_str = "attn OOM"
                lat_r_str = "—"
            md.append(f"| {img} | {nv} | {T} | {cell(attn)} | {cell(ssd_pt)} | {cell(ssd_tr)} | {mem_r_str} | {lat_r_str} |\n")
    md.append("\nMem/lat ratios = Triton-kernel Mamba-3 / DA3 transformer (lower is better for both).\n")
    (out_dir / "efficiency_table.md").write_text("".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[bench] matplotlib not available; skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels_to_plot = [
        ("DA3-SMALL (transformer)", "o-", "tab:blue"),
        ("DA3-SMALL +Mamba-3 (PyTorch)", "s--", "tab:orange"),
        ("DA3-SMALL +Mamba-3 (Triton kernel)", "^-", "tab:green"),
    ]
    for img in sizes:
        for label, marker, color in labels_to_plot:
            xs, lats, mems = [], [], []
            for nv in n_views_list:
                T = nv * (img // 14) ** 2
                r = by_key.get((img, nv, label))
                if r is None or r.get("OOM"):
                    continue
                xs.append(T)
                lats.append(r["latency_ms"])
                mems.append(r["peak_MiB"])
            if xs:
                axes[0].plot(xs, lats, marker, color=color, label=f"{label} (img={img}²)")
                axes[1].plot(xs, mems, marker, color=color, label=f"{label} (img={img}²)")
    axes[0].set_xlabel("cross-view sequence length T")
    axes[0].set_ylabel("latency (ms)")
    axes[0].set_title("Forward latency vs sequence length")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("cross-view sequence length T")
    axes[1].set_ylabel("peak GPU memory (MiB)")
    axes[1].set_title("Peak GPU memory vs sequence length")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plot_path = out_dir / "efficiency_comparison.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"\n[bench] wrote {out_dir}/efficiency_table.md + efficiency.json + efficiency_comparison.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 392, 504])
    ap.add_argument("--n-views", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="If set, write efficiency_table.md, efficiency.json, "
                    "and efficiency_comparison.png under this directory.")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"\n[bench_efficiency_patched] DA3 full-model forward, B=1\n")

    rows = run_grid(args.sizes, args.n_views, args.state_dim, device)
    summary_ratios(rows, args.sizes, args.n_views)
    if args.out_dir is not None:
        write_outputs(rows, args.sizes, args.n_views, args.out_dir)


if __name__ == "__main__":
    main()
