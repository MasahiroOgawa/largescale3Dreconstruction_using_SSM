#!/usr/bin/env python3
"""Per-pixel depth metrics on ETH3D `terrains` for a trained DA3 / Vision-Mamba-3 ckpt.

Why this exists rather than the DA3 bench evaluator: on this scene the reconstruction
metrics are degenerate. F-score@5cm came out 0.0000-0.0005 for every arm here, and
doc/PLAN.md records the same for un-patched zero-shot DA3 (F_posed=0.0001). `terrains` is
outdoor at tens of metres, so a 5 cm surface threshold is unreachable and every method
scores zero -- the metric cannot separate them. The CM series used depth metrics for
exactly this reason, and the evaluator that produced them
(scripts/eval_ssm3d_vs_da3.py, deleted in 540a700) predates the current swap architecture
and cannot load these checkpoints.

Metrics are the CM table's: |relative_depth_error|, delta<1.25, RMSE and log10, all after
per-image median scale alignment, which is what makes a monocular prediction comparable to
metric ground truth at all.

  uv run python scripts/eval_depth_metrics.py --ckpt result/runs/depth_ft_vssd_bg/ckpt_1000.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mamba3_attn.data.bench import load_bench_scene
from mamba3_attn.eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from mamba3_attn.eval.metrics import (
    abs_relative_depth_error,
    align_scale_median,
    delta_threshold,
    log10_metric,
    rmse,
)
from mamba3_attn.patch import install_mamba3
from mamba3_attn.train.da3_loss import DA3LossWeights
from mamba3_attn.train.train_super import SuperPhaseConfig

# Allowlist the one class our own checkpoints store, so the weights can be read with
# weights_only=True. torch.load's default (False) unpickles arbitrary objects and would
# execute code from any file handed to it; allowlisting is the narrow fix, and these
# checkpoints do contain a dataclass (`cfg`) that the safe unpickler otherwise rejects.
torch.serialization.add_safe_globals([SuperPhaseConfig, DA3LossWeights])


def build(ckpt: Path | None, device: str, state_dim: int = 64):
    """Rebuild the student exactly as it was trained, then load its weights.

    variant and rope_all_layers come from the checkpoint's own cfg rather than from flags.
    RoPE holds no parameters, so a mismatch here would load cleanly and silently evaluate a
    different model than was trained -- blocks 0-3 unencoded at eval, encoded at train.
    """
    if ckpt is None:
        api = load_da3(DEFAULT_HF_MODEL, device=device)
        api.model.eval()
        return api, {"variant": None, "rope_all_layers": False}

    state = torch.load(ckpt, map_location=device, weights_only=True)
    cfg = state.get("cfg") or {}
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    variant = cfg.get("variant", "mamba3")
    rope_all = bool(cfg.get("rope_all_layers", False))
    # The baseline arm was trained with --no-mamba3-swap, and its cfg still carries the
    # default variant. Patching on that would build a model the checkpoint never had.
    swapped = not bool(cfg.get("no_mamba3_swap", False))

    api = load_da3(DEFAULT_HF_MODEL, device=device)
    if swapped:
        install_mamba3(api.model, which="all", variant=variant, state_dim=state_dim,
                       use_fused_kernel=False, chunk_size=128, rope_all_layers=rope_all)
    else:
        variant, rope_all = None, False
    api.model.load_state_dict(state["model"])
    api.model.eval()
    return api, {"variant": variant, "rope_all_layers": rope_all}


@torch.no_grad()
def evaluate(api, sample, device: str, image_size: int = 504) -> dict:
    """Median-aligned depth metrics, averaged over images.

    Aligned and scored per image rather than over the pooled tensor: a single frame with a
    large scale offset would otherwise dominate a global alignment and drag every frame's
    error with it.
    """
    # Paths, not tensors: DA3's inference runs its own loading and reference-view
    # selection, and `saddle_balanced` is the strategy the CM numbers were measured with.
    pred = api.inference([str(p) for p in sample.image_paths], process_res=image_size,
                         ref_view_strategy="saddle_balanced", export_dir=None)
    depth = torch.from_numpy(np.asarray(pred.depth)).float().squeeze()

    gt, valid = sample.gt_depth, sample.valid_mask
    if depth.shape[-2:] != gt.shape[-2:]:
        # DA3 predicts at its own processing resolution; score against GT at GT's size so
        # no ground-truth detail is resampled away.
        depth = F.interpolate(depth[:, None], size=gt.shape[-2:],
                              mode="bilinear", align_corners=False)[:, 0]
    if valid is None:
        valid = torch.isfinite(gt) & (gt > 0)

    acc: dict[str, list[float]] = {k: [] for k in
                                   ("abs_rel", "delta_1_25", "rmse", "log10")}
    for i in range(gt.shape[0]):
        v = valid[i]
        if not bool(v.any()):
            continue
        p = align_scale_median(depth[i], gt[i], v)
        acc["abs_rel"].append(abs_relative_depth_error(p, gt[i], v))
        acc["delta_1_25"].append(delta_threshold(p, gt[i], v, 1.25))
        acc["rmse"].append(rmse(p, gt[i], v))
        acc["log10"].append(log10_metric(p, gt[i], v))

    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()} | {
        "n_images": len(acc["abs_rel"])
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="omit to evaluate un-patched DA3-SMALL as published")
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--scene", type=str, default="terrains")
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    api, meta = build(args.ckpt, args.device)
    sample = load_bench_scene("eth3d", args.scene, args.data_root,
                              max_images=args.max_images, image_size=args.image_size,
                              load_gt_depth=True)
    m = evaluate(api, sample, args.device, args.image_size)
    label = args.label or (args.ckpt.parent.name if args.ckpt else "DA3-SMALL")
    row = {"label": label, **meta, **m}
    print(f"{label:24s} abs_rel={m['abs_rel']:.4f}  d<1.25={m['delta_1_25']:.4f}  "
          f"rmse={m['rmse']:.4f}  log10={m['log10']:.4f}  (n={m['n_images']})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
