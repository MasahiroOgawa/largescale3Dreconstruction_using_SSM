"""Merge `run_vit_attn.log` into `results.json` after a `--variants vit_mamba3`
resume overwrote it (see `doc/PLAN.md §9.10.1`). Re-renders summary.md and figures.

Usage:
    uv run python scripts/merge_vit_attn_log_into_results.py \
        --out-dir outputs/cifar10_compare_patch1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_cifar10_compare import make_all_figures  # noqa: E402

EPOCH_RE = re.compile(
    r"^\s+ep\s+(\d+)\s+trL\s+(\S+)\s+trA\s+(\S+)\s+teL\s+(\S+)\s+teA\s+(\S+)\s+lr\s+(\S+)\s+(\S+)s",
    re.MULTILINE,
)
PARAMS_RE = re.compile(r"params:\s+\S+\s+M\s+\(([\d,]+)\)")
EFF_RE = re.compile(r"efficiency:\s+latency\s+(\S+)\s+ms\s+peak\s+(\S+)\s+MiB")

VARIANT_ORDER = ("cnn", "vit_attn", "vit_mamba3")


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text()
    epochs = [
        {
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "train_acc": float(m.group(3)) / 100.0,
            "test_loss": float(m.group(4)),
            "test_acc": float(m.group(5)) / 100.0,
            "lr_end": float(m.group(6)),
            "train_wall_s": float(m.group(7)),
        }
        for m in EPOCH_RE.finditer(text)
    ]
    if not epochs:
        raise ValueError(f"no epoch lines parsed from {log_path}")

    params_match = PARAMS_RE.search(text)
    eff_match = EFF_RE.search(text)
    if not params_match or not eff_match:
        raise ValueError(f"params/efficiency line missing in {log_path}")

    return {
        "variant": "vit_attn",
        "params": int(params_match.group(1).replace(",", "")),
        "best_test_acc": max(e["test_acc"] for e in epochs),
        "final_train_acc": epochs[-1]["train_acc"],
        "final_test_acc": epochs[-1]["test_acc"],
        "mean_train_wall_s": sum(e["train_wall_s"] for e in epochs) / len(epochs),
        "efficiency": {
            "latency_ms": float(eff_match.group(1)),
            "peak_mib": float(eff_match.group(2)),
        },
        "epochs": epochs,
    }


def write_summary_md(out: Path, results: dict, cfg: dict) -> None:
    label = {"cnn": "CNN (small ResNet)", "vit_attn": "ViT-Tiny + softmax",
             "vit_mamba3": "ViT-Tiny + Mamba-3 SSD"}
    grad_clip = cfg.get("grad_clip", 0)
    lines = [
        "# CIFAR-10 compare — summary",
        "",
        f"Recipe: AdamW lr={cfg['lr']}, wd={cfg.get('weight_decay', 0.05)}, "
        f"{cfg['warmup_epochs']}-ep warmup → {cfg['lr_schedule']}, "
        f"grad_clip={grad_clip if grad_clip > 0 else 'off'}, "
        f"batch={cfg['batch_size']}, epochs={cfg['epochs']}, "
        f"seed={cfg['seed']}, device={cfg['device']}, patch_size={cfg['patch_size']}.",
        "",
        "## Head-to-head",
        "",
        "| Variant | Params (M) | Train Acc | Test Acc | Train s/epoch | Test lat (ms, B=128) | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        if variant not in results:
            continue
        r = results[variant]
        lines.append(
            f"| {label[variant]} | {r['params']/1e6:.2f} | "
            f"{r['final_train_acc']*100:.2f} | {r['best_test_acc']*100:.2f} | "
            f"{r['mean_train_wall_s']:.1f} | {r['efficiency']['latency_ms']:.2f} | "
            f"{r['efficiency']['peak_mib']:.1f} |"
        )

    a = results.get("vit_attn")
    m = results.get("vit_mamba3")
    lines += ["", "## Acceptance gate (`PLAN.md §5`, T=1025 efficiency context)", ""]
    if a is None or m is None:
        lines.append("_Not evaluable_: gate requires both vit_attn and vit_mamba3 results.")
    else:
        acc_gap_pp = (m["best_test_acc"] - a["best_test_acc"]) * 100.0
        mem_ratio = m["efficiency"]["peak_mib"] / a["efficiency"]["peak_mib"]
        lat_ratio = m["efficiency"]["latency_ms"] / a["efficiency"]["latency_ms"]
        acc_pass = acc_gap_pp >= -2.0
        mem_pass = mem_ratio <= 1.1
        verdict = "PASS" if (acc_pass and mem_pass) else "FAIL"
        lines += [
            f"- Acc gap (mamba3 − softmax): **{acc_gap_pp:+.2f} pp** "
            f"(threshold ≥ −2.00 pp) → {'PASS' if acc_pass else 'FAIL'}",
            f"- Mem ratio (mamba3 / softmax): **{mem_ratio:.2f}×** "
            f"(threshold ≤ 1.10×) → {'PASS' if mem_pass else 'FAIL'}",
            f"- Latency ratio (mamba3 / softmax): **{lat_ratio:.2f}×** (informational)",
            "",
            f"**Verdict: {verdict}**",
            "",
            "Note: at `patch_size=1` (T=1025) the headline is efficiency, not the §5 "
            "accuracy gate (which was framed for T=65). Both variants are undertrained "
            "at 30 ep; treat accuracy as orientation only.",
        ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def write_efficiency_table_md(out: Path, results: dict) -> None:
    lines = [
        "# Efficiency table — CIFAR-10 compare (B=128, model T per `patch_size`)",
        "",
        "| Variant | Params (M) | Latency (ms) | Peak (MiB) | Train s/epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        if variant not in results:
            continue
        r = results[variant]
        lines.append(
            f"| {variant} | {r['params']/1e6:.2f} | "
            f"{r['efficiency']['latency_ms']:.2f} | "
            f"{r['efficiency']['peak_mib']:.1f} | "
            f"{r['mean_train_wall_s']:.1f} |"
        )
    (out / "efficiency_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--vit-attn-log", type=Path, default=None,
                    help="default: <out-dir>/run_vit_attn.log")
    args = ap.parse_args()

    out = args.out_dir
    log_path = args.vit_attn_log or (out / "run_vit_attn.log")
    results_path = out / "results.json"

    payload = json.loads(results_path.read_text())
    if "vit_attn" in payload["variants"]:
        print(f"[merge] vit_attn already in {results_path}; replacing.")
    payload["variants"]["vit_attn"] = parse_log(log_path)

    payload["variants"] = {k: payload["variants"][k]
                           for k in VARIANT_ORDER
                           if k in payload["variants"]}

    results_path.write_text(json.dumps(payload, indent=2))
    print(f"[merge] wrote {results_path}")

    write_summary_md(out, payload["variants"], payload["config"])
    write_efficiency_table_md(out, payload["variants"])
    print(f"[merge] wrote {out / 'summary.md'} and {out / 'efficiency_table.md'}")

    for p in make_all_figures(results_path, out):
        print(f"[merge] figure → {p}")


if __name__ == "__main__":
    main()
