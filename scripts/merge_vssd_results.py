"""Merge CIFAR-10 per-recipe results.json files into a single 4-variant payload
and regenerate the comparison plots via plot_cifar10_compare.make_all_figures.

Used by `doc/PLAN_cifar10.md §9.x` to land VSSD alongside the prior CNN /
ViT-softmax / Mamba-3-SSD baselines without re-training them.

Usage:
    uv run python scripts/merge_vssd_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cifar10_compare import make_all_figures  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"

# Core hyperparameter keys that must match across runs being merged.
# Newer runs include extra config keys (lr_schedule, steps_per_epoch, patch_size,
# mamba_chunk_size, plateau_*) — we ignore those when checking compatibility
# so legacy runs (pre-dating those flags) can still be merged.
CORE = ("epochs", "batch_size", "lr", "weight_decay", "seed", "warmup_epochs", "grad_clip")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_core(cfgs: list[dict], label: str) -> None:
    a = cfgs[0]
    for b in cfgs[1:]:
        for k in CORE:
            if k in a and k in b and a[k] != b[k]:
                raise AssertionError(f"[{label}] core key {k!r} mismatch: {a[k]!r} vs {b[k]!r}")


def merge(name: str, sources: dict[str, Path], dest_dir: Path) -> None:
    """Combine variants from multiple source results.json files into one.

    `sources` maps {variant_key: results.json_path}. The variant under that key
    must exist in the source's `variants` dict; we copy it into the merged payload.
    The merged `config` is taken from the first source whose config is most complete.
    """
    payloads = {v: _load(p) for v, p in sources.items()}
    _check_core([p["config"] for p in payloads.values()], name)

    # Pick the most-complete config (highest number of keys) as base.
    base_cfg = max((p["config"] for p in payloads.values()), key=len)
    merged_variants: dict[str, dict] = {}
    for v, p in payloads.items():
        if v not in p["variants"]:
            raise KeyError(f"[{name}] variant {v!r} missing in {p}")
        merged_variants[v] = p["variants"][v]

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "results.json"
    out_path.write_text(json.dumps({"config": base_cfg, "variants": merged_variants}, indent=2))
    print(f"[{name}] merged {len(merged_variants)} variants → {out_path}")
    figs = make_all_figures(out_path, dest_dir)
    for p in figs:
        print(f"  figure → {p}")


def main() -> None:
    merge(
        "cosine_50ep",
        {
            "cnn":              OUT / "cifar10_compare" / "results.json",
            "vit_attn":         OUT / "cifar10_compare" / "results.json",
            "vit_mamba3":       OUT / "cifar10_compare" / "results.json",
            "vit_mamba3_vssd":  OUT / "cifar10_compare_vssd_cosine" / "results.json",
        },
        OUT / "cifar10_compare_cosine_with_vssd",
    )
    merge(
        "mamba_recipe_80ep",
        {
            "cnn":              OUT / "cifar10_compare_mamba_recipe_topup" / "results.json",
            "vit_attn":         OUT / "cifar10_compare_mamba_recipe_topup" / "results.json",
            "vit_mamba3":       OUT / "cifar10_compare_mamba_recipe" / "results.json",
            "vit_mamba3_vssd":  OUT / "cifar10_compare_vssd_mamba_recipe" / "results.json",
        },
        OUT / "cifar10_compare_mamba_recipe_with_vssd",
    )
    merge(
        "plateau_80ep",
        {
            "cnn":              OUT / "cifar10_compare_plateau" / "results.json",
            "vit_attn":         OUT / "cifar10_compare_plateau" / "results.json",
            "vit_mamba3":       OUT / "cifar10_compare_plateau" / "results.json",
            "vit_mamba3_vssd":  OUT / "cifar10_compare_vssd_plateau" / "results.json",
        },
        OUT / "cifar10_compare_plateau_with_vssd",
    )


if __name__ == "__main__":
    main()
