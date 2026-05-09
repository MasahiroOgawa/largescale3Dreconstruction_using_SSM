"""Test-loss diagnostic for the per-layer scene-overfit ablation (PLAN §15.59.6).

For each `outputs/runs/per_layer_overfit/layer_KK/ckpt_<step>.pt`, compute the
DA3 paper loss (same form used during training) on **train** views and on
held-out **test** views. Outputs:

  - `<out>/test_loss.json`        — per-ckpt train/test loss + components.
  - `<out>/test_loss_summary.md`  — markdown table per layer × ckpt.
  - `<out>/test_loss_curves.png`  — train vs test total loss per ckpt step.

The diagnostic disambiguates two failure modes that the AUC@30°/F-score eval
can't distinguish:
  (a) memorization — train loss ↓ but test loss flat/↑.
  (b) noisy AUC/F  — both losses ↓ but downstream metric is high-variance on
                      a 10-view test split.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mamba3_attn.eval.phase4_evaluator import build_patched_api
from mamba3_attn.train.da3_loss import DA3LossWeights, da3_paper_loss
from mamba3_attn.train.multi_view import _slice_batch, load_full_scene_cache
from mamba3_attn.train.train_super import _student_forward, build_target


CKPT_RE = re.compile(r"ckpt_(\d+)\.pt$")
EVAL_LOG_RE = re.compile(r"eval_ckpt_(\d+)\.log$")
MEAN_LINE_RE = re.compile(
    r"^MEAN\s+([-\d.eE+nan]+)\s+([-\d.eE+nan]+)\s+([-\d.eE+nan]+)\s+([-\d.eE+nan]+)\s*$"
)


def _read_auc30_per_step(layer_dir: Path) -> dict[int, float]:
    """Parse layer_KK/eval_ckpt_*.log -> {step: auc30}, ignoring missing/corrupt logs."""
    out: dict[int, float] = {}
    for log in layer_dir.glob("eval_ckpt_*.log"):
        m = EVAL_LOG_RE.search(log.name)
        if not m:
            continue
        step = int(m.group(1))
        for line in log.read_text(errors="ignore").splitlines():
            mm = MEAN_LINE_RE.match(line)
            if mm:
                try:
                    out[step] = float(mm.group(1))
                except ValueError:
                    pass
                break
    return out


def _ckpt_step(path: Path) -> int | None:
    m = CKPT_RE.search(path.name)
    return int(m.group(1)) if m else None


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


@torch.no_grad()
def _measure_loss(
    api, cache, indices: list[int], n_views: int, n_batches: int,
    image_size: int, device, weights: DA3LossWeights, seed: int,
    cam_posed: bool, amp_dtype: torch.dtype,
) -> dict:
    """Average DA3 paper loss + components over `n_batches` random subsets."""
    rng = random.Random(seed)
    totals: list[float] = []
    comps: dict[str, list[float]] = {k: [] for k in ("L_D", "L_M", "L_grad", "L_P", "L_C")}
    for _ in range(n_batches):
        picked = rng.sample(indices, n_views)
        batch = _slice_batch(cache, picked)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
            target, gt_kwargs, _ = build_target(batch, None, image_size, device)
            student_extr = gt_kwargs.get("gt_w2c") if cam_posed else None
            student_intr = gt_kwargs.get("gt_intrinsics") if cam_posed else None
            s_out = _student_forward(
                api, batch.images.to(device),
                extrinsics=student_extr, intrinsics=student_intr,
            )
            loss_out = da3_paper_loss(student=s_out, target=target, weights=weights, **gt_kwargs)
        totals.append(float(loss_out.total))
        comps["L_D"].append(float(loss_out.l_depth))
        comps["L_M"].append(float(loss_out.l_ray))
        comps["L_grad"].append(float(loss_out.l_grad))
        comps["L_P"].append(float(loss_out.l_point))
        comps["L_C"].append(float(loss_out.l_cam))
    return {"total": _mean(totals), **{k: _mean(v) for k, v in comps.items()}}


def _layer_idx(d: Path) -> int | None:
    try:
        return int(d.name.split("_")[1])
    except (ValueError, IndexError):
        return None


def _layer_uses_cam_enc(layer_k: int, n_backbone_layers: int) -> bool:
    return layer_k >= n_backbone_layers


def _plot_curves(results: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    cmap = plt.cm.viridis
    items = sorted(results.items())
    n = max(len(items), 1)
    for split_name, ax in zip(("train", "test"), axes):
        for i, (k, ckpts) in enumerate(items):
            steps = sorted(int(s) for s in ckpts)
            ys = [ckpts[str(s)][split_name]["total"] for s in steps]
            color = cmap(i / max(n - 1, 1))
            ax.plot(steps, ys, "o-", label=f"layer {k:02d}", color=color,
                    linewidth=1.2, markersize=4, alpha=0.85)
        ax.set_xlabel("ckpt step")
        ax.set_ylabel(f"{split_name} loss (DA3 paper, mean)")
        ax.set_title(f"{split_name} loss vs ckpt step (per swapped layer)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2, loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_summary(results: dict, out_path: Path, run_dir: Path) -> None:
    lines = [
        "# Per-layer test-loss diagnostic (PLAN §15.59.6)\n",
        "\nDA3 paper loss measured on training views and held-out test views\n"
        "(same scene, same split as training). Means over randomly sampled\n"
        "n_views=4 batches; no augmentation.\n",
        "\n## Total loss per ckpt\n",
        "\n| Layer | ckpt | train_loss | test_loss | gap (test−train) |\n",
        "|---|---|---|---|---|\n",
    ]
    for k, ckpts in sorted(results.items()):
        for s in sorted(int(x) for x in ckpts):
            tr = ckpts[str(s)]["train"]["total"]
            te = ckpts[str(s)]["test"]["total"]
            lines.append(
                f"| {k:02d} | {s} | {tr:.4f} | {te:.4f} | {te - tr:+.4f} |\n"
            )

    lines.append(
        "\n## Test-loss vs AUC@30° alignment\n"
        "\nPer-layer comparison of the ckpt that minimizes test-loss vs the ckpt that\n"
        "maximizes AUC@30°. If the two agree, training-objective improvement tracks the\n"
        "downstream pose-AUC metric; if they disagree, the training objective and the\n"
        "eval metric are misaligned.\n"
        "\n| Layer | best_test_loss step | min test_loss | AUC@30° peak step | peak AUC@30° | aligned? |\n"
        "|---|---|---|---|---|---|\n"
    )
    aligned_count = 0
    n_layers = 0
    for k, ckpts in sorted(results.items()):
        n_layers += 1
        best_test_step = min(ckpts, key=lambda s: ckpts[s]["test"]["total"])
        min_test_loss = ckpts[best_test_step]["test"]["total"]
        auc_per_step = _read_auc30_per_step(run_dir / f"layer_{k:02d}")
        if auc_per_step:
            auc_peak_step = max(auc_per_step, key=lambda s: auc_per_step[s])
            auc_peak = auc_per_step[auc_peak_step]
            same = int(best_test_step) == auc_peak_step
            aligned_count += int(same)
            mark = "✓" if same else "✗"
            lines.append(
                f"| {k:02d} | {best_test_step} | {min_test_loss:.4f} "
                f"| {auc_peak_step} | {auc_peak:.4f} | {mark} |\n"
            )
        else:
            lines.append(
                f"| {k:02d} | {best_test_step} | {min_test_loss:.4f} | n/a | n/a | n/a |\n"
            )
    lines.append(f"\nAligned layers: {aligned_count} / {n_layers}.\n")
    out_path.write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="Same dir as scripts/per_layer_overfit.py.")
    ap.add_argument("--scene", default="terrains")
    ap.add_argument("--dataset", default="eth3d", choices=["eth3d", "hiroom", "7scenes"])
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--candidate-views", type=int, default=256)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--n-batches", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--n-backbone-layers", type=int, default=12)
    ap.add_argument("--layers", type=str, default=None,
                    help="Comma-separated subset (default: all layer_* dirs found).")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip GPU evaluation; just re-render summary + plot from "
                    "an existing test_loss.json.")
    args = ap.parse_args()

    if args.aggregate_only:
        json_path = args.out / "test_loss.json"
        raw = json.loads(json_path.read_text())
        results = {int(k): v for k, v in raw.items()}
        _write_summary(results, args.out / "test_loss_summary.md", args.out)
        _plot_curves(results, args.out / "test_loss_curves.png")
        print(f"wrote {args.out / 'test_loss_summary.md'} (aggregate-only)", flush=True)
        return

    device = torch.device(args.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp_dtype]
    weights = DA3LossWeights()
    weights.use_kendall_gal = False  # matches §15.59.3 revert

    print(f"[test-loss] loading scene cache {args.dataset}/{args.scene}", flush=True)
    cache = load_full_scene_cache(
        args.dataset, args.scene, Path("data"),
        image_size=args.image_size, candidate_views=args.candidate_views,
        frame_stride=args.frame_stride,
    )

    layer_dirs = [d for d in sorted(args.out.glob("layer_*")) if d.is_dir()]
    if args.layers:
        wanted = {int(x) for x in args.layers.split(",")}
        layer_dirs = [d for d in layer_dirs if _layer_idx(d) in wanted]

    # Merge with any pre-existing test_loss.json so a partial re-run (e.g.
    # --layers 12,13,14,15) doesn't wipe out earlier layers' data.
    json_path = args.out / "test_loss.json"
    results: dict = {}
    if json_path.exists():
        prior = json.loads(json_path.read_text())
        results = {int(k): v for k, v in prior.items()}
    for ld in layer_dirs:
        k = _layer_idx(ld)
        if k is None:
            continue
        split_path = ld / "split.json"
        if not split_path.exists():
            print(f"[test-loss] layer {k:02d}: no split.json, skipping", flush=True)
            continue
        split = json.loads(split_path.read_text())
        train_idx = list(split["train"])
        test_idx = list(split["test"])
        cam_posed = _layer_uses_cam_enc(k, args.n_backbone_layers)

        ckpts = sorted(ld.glob("ckpt_*.pt"), key=lambda p: _ckpt_step(p) or -1)
        if not ckpts:
            print(f"[test-loss] layer {k:02d}: no ckpts, skipping", flush=True)
            continue

        per_layer: dict = {}
        for ckpt in ckpts:
            step = _ckpt_step(ckpt)
            if step is None:
                continue
            print(f"[test-loss] layer {k:02d} ckpt {step}", flush=True)
            api = build_patched_api(
                str(ckpt), device=args.device, state_dim=args.state_dim,
                patched=True, swap_layers=[k],
            )
            train_m = _measure_loss(
                api, cache, train_idx, args.n_views, args.n_batches,
                args.image_size, device, weights, args.seed,
                cam_posed=cam_posed, amp_dtype=amp_dtype,
            )
            test_m = _measure_loss(
                api, cache, test_idx, args.n_views, args.n_batches,
                args.image_size, device, weights, args.seed + 1,
                cam_posed=cam_posed, amp_dtype=amp_dtype,
            )
            per_layer[str(step)] = {"train": train_m, "test": test_m, "cam_posed": cam_posed}
            del api
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results[k] = per_layer

    json_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {json_path}", flush=True)

    _write_summary(results, args.out / "test_loss_summary.md", args.out)
    print(f"wrote {args.out / 'test_loss_summary.md'}", flush=True)

    _plot_curves(results, args.out / "test_loss_curves.png")
    print(f"wrote {args.out / 'test_loss_curves.png'}", flush=True)


if __name__ == "__main__":
    main()
