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


def build_da3_patched(device: str, state_dim: int, variant: str = "mamba3",
                      use_fused_kernel: bool = True):
    api = load_da3(DEFAULT_HF_MODEL, device=device)
    install_mamba3(
        api.model, which="all", variant=variant, state_dim=state_dim,
        use_fused_kernel=use_fused_kernel, chunk_size=128,
    )
    return api.to(device)


# Variant spec: cli_name → (display_label, build_fn_factory).
# build_fn_factory(device_type, state_dim) returns a zero-arg callable that
# constructs the api. Keeping it lazy lets measure_variant build, measure, and
# free each model before the next variant constructs.
VARIANT_SPECS: dict[str, tuple[str, callable]] = {
    "da3_small": (
        "DA3-SMALL (transformer)",
        lambda dev, sd: (lambda: build_da3_unpatched(dev)),
    ),
    "mamba3_pt": (
        "DA3-SMALL +Mamba-3 (PyTorch)",
        lambda dev, sd: (lambda: build_da3_patched(dev, sd, variant="mamba3", use_fused_kernel=False)),
    ),
    "mamba3_triton": (
        "DA3-SMALL +Mamba-3 (Triton kernel)",
        lambda dev, sd: (lambda: build_da3_patched(dev, sd, variant="mamba3", use_fused_kernel=True)),
    ),
    "vssd": (
        "DA3-SMALL +VSSD (NC-SSD)",
        lambda dev, sd: (lambda: build_da3_patched(dev, sd, variant="vssd", use_fused_kernel=False)),
    ),
}

DEFAULT_VARIANTS = ["da3_small", "mamba3_pt", "mamba3_triton", "vssd"]


def run_grid(sizes: list[int], n_views_list: list[int], state_dim: int,
             device: torch.device, variants: list[str]) -> list[dict]:
    rows: list[dict] = []
    for img_size in sizes:
        for n_views in n_views_list:
            x = torch.randn(1, n_views, 3, img_size, img_size, device=device)
            tokens_per_view = (img_size // 14) ** 2
            T_total = n_views * tokens_per_view
            print(f"\n[bench] img={img_size}² S={n_views}  tokens/view={tokens_per_view}  cross-view T={T_total}")

            for name in variants:
                label, factory = VARIANT_SPECS[name]
                r = measure_variant(label, factory(device.type, state_dim), x, device)
                r["tokens_per_view"] = tokens_per_view
                r["cross_view_T"] = T_total
                r["variant"] = name
                rows.append(r)
                _print_row(r)

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


def summary_ratios(rows: list[dict], sizes: list[int], n_views_list: list[int],
                   variants: list[str]) -> None:
    by_key: dict[tuple[int, int, str], dict] = {}
    for r in rows:
        by_key[(r["img_size"], r["n_views"], r["label"])] = r

    baseline_label = VARIANT_SPECS["da3_small"][0]
    challengers = [v for v in variants if v != "da3_small"]

    print("\n" + "=" * 80)
    print(f"Apples-to-apples: <variant> / DA3-SMALL (transformer) — mem ratio · lat ratio")
    print("=" * 80)
    head_cols = "".join(f"{VARIANT_SPECS[v][0][:22]:>24}" for v in challengers)
    print(f"{'img':>5} {'S':>3} {'cross-T':>8} {head_cols}")
    print("-" * (17 + 24 * len(challengers)))
    for img in sizes:
        for nv in n_views_list:
            attn = by_key.get((img, nv, baseline_label))
            if not attn:
                continue
            T = nv * (img // 14) ** 2
            cells: list[str] = []
            for v in challengers:
                row = by_key.get((img, nv, VARIANT_SPECS[v][0]))
                if not row:
                    cells.append(f"{'-':>24}")
                elif attn.get("OOM") and row.get("OOM"):
                    cells.append(f"{'both OOM':>24}")
                elif attn.get("OOM"):
                    cells.append(f"{'attn OOM, ok':>24}")
                elif row.get("OOM"):
                    cells.append(f"{'OOM':>24}")
                else:
                    mem_r = row["peak_MiB"] / attn["peak_MiB"]
                    lat_r = row["latency_ms"] / attn["latency_ms"]
                    cells.append(f"{mem_r:>10.2f}× {lat_r:>10.2f}×")
            print(f"{img:>5} {nv:>3} {T:>8}{''.join(cells)}")


_VARIANT_STYLE: dict[str, tuple[str, str]] = {
    "da3_small": ("o-", "tab:blue"),
    "mamba3_pt": ("s--", "tab:orange"),
    "mamba3_triton": ("^-", "tab:green"),
    "vssd": ("D-", "tab:red"),
}


def write_outputs(rows: list[dict], sizes: list[int], n_views_list: list[int],
                  variants: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "efficiency.json").write_text(json.dumps(rows, indent=2))

    by_key: dict[tuple[int, int, str], dict] = {
        (r["img_size"], r["n_views"], r["label"]): r for r in rows
    }
    baseline_label = VARIANT_SPECS["da3_small"][0]
    challengers = [v for v in variants if v != "da3_small"]

    def cell(r: dict | None) -> str:
        if r is None:
            return "-"
        if r.get("OOM"):
            return "**OOM**"
        return f"{r['peak_MiB']:.0f} MiB / {r['latency_ms']:.0f} ms"

    md = ["# Efficiency comparison — DA3-SMALL transformer vs Mamba-3 / VSSD\n",
          "\nFull-model forward (backbone + DPT head + cam_dec), B=1, GPU.\n",
          "Apples-to-apples: same architecture, only attention modules differ.\n\n"]
    header = ["img", "S", "cross-T"] + [VARIANT_SPECS[v][0] for v in variants]
    header += [f"mem×({v})" for v in challengers] + [f"lat×({v})" for v in challengers]
    md.append("| " + " | ".join(header) + " |\n")
    md.append("|" + "|".join("---" for _ in header) + "|\n")
    for img in sizes:
        for nv in n_views_list:
            T = nv * (img // 14) ** 2
            baseline = by_key.get((img, nv, baseline_label))
            cells = [str(img), str(nv), str(T)]
            cells += [cell(by_key.get((img, nv, VARIANT_SPECS[v][0]))) for v in variants]
            for v in challengers:
                row = by_key.get((img, nv, VARIANT_SPECS[v][0]))
                if not baseline or not row or baseline.get("OOM") or row.get("OOM"):
                    cells.append("—")
                else:
                    cells.append(f"{row['peak_MiB'] / baseline['peak_MiB']:.2f}×")
            for v in challengers:
                row = by_key.get((img, nv, VARIANT_SPECS[v][0]))
                if not baseline or not row or baseline.get("OOM") or row.get("OOM"):
                    cells.append("—")
                else:
                    cells.append(f"{row['latency_ms'] / baseline['latency_ms']:.2f}×")
            md.append("| " + " | ".join(cells) + " |\n")
    md.append("\nmem×/lat× ratios = <variant> / DA3-SMALL (transformer); lower is better.\n")
    (out_dir / "efficiency_table.md").write_text("".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[bench] matplotlib not available; skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels_to_plot = [(VARIANT_SPECS[v][0], *_VARIANT_STYLE[v]) for v in variants]
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
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                    choices=list(VARIANT_SPECS),
                    help="Subset of variants to benchmark. Order is preserved in tables/plots.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="If set, write efficiency_table.md, efficiency.json, "
                    "and efficiency_comparison.png under this directory.")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"\n[bench_efficiency_patched] DA3 full-model forward, B=1, variants={args.variants}\n")

    rows = run_grid(args.sizes, args.n_views, args.state_dim, device, args.variants)
    summary_ratios(rows, args.sizes, args.n_views, args.variants)
    if args.out_dir is not None:
        write_outputs(rows, args.sizes, args.n_views, args.variants, args.out_dir)


if __name__ == "__main__":
    main()
