"""Run DA3-metric on the same val clips used by the scale-estimator and
measure pooled / per-subset MAE for the per-clip scene-scale metric
(`s = median anchor-frame Z` over query tracks).

Purpose: confirm whether DA3-metric on its own ALREADY does what our
standalone scale estimator was trying to learn. If yes, the standalone
trained model (v4 best 1.79 m pooled MAE) is redundant, and the right
next step is to use DA3-metric as a frozen module in the joint tracker.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# DA3's api.py imports moviepy.editor inside the GS-export module — moviepy
# 2.x dropped that submodule. Stub it so we can import the API; we never use
# the GS export path.
sys.modules.setdefault("moviepy.editor", types.ModuleType("moviepy.editor"))
sys.path.insert(0, "third_party/depth-anything-3/src")

from depth_anything_3.api import DepthAnything3  # noqa: E402

# Local: dataset split + clip subset helper.
from mamba3_tracker.data.dataset import official_train_test_split  # noqa: E402


def _which_subset(path: Path) -> str:
    p = str(path)
    for s in ("pstudio", "drivetrack", "adt"):
        if f"/{s}/" in p:
            return s
    return "unknown"


def _load_anchor_frame_and_queries(clip_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rgb_anchor HxWx3 uint8, queries Nx3 float32 (x,y,t), gt_z_at_anchor N float32).

    The "anchor" frame is the median anchor-frame index across queries — we
    use a single fixed anchor for simplicity. Queries that anchor at other
    frames still contribute their depth at THIS anchor frame's GT XYZ.
    Per-clip scale is then median over queries of Z at that anchor frame.
    """
    d = np.load(clip_path, allow_pickle=True)
    jpegs = d["images_jpeg_bytes"]
    queries = d["queries_xyt"].astype(np.float32)        # (N, 3)
    tracks  = d["tracks_XYZ"].astype(np.float32)         # (F, N, 3)
    # Pick the median anchor frame index across queries as the "shared" anchor.
    anchor_t = int(np.median(queries[:, 2]))
    rgb = np.asarray(Image.open(io.BytesIO(jpegs[anchor_t])).convert("RGB"))  # (H, W, 3) uint8
    gt_z_anchor = tracks[anchor_t, :, 2]                # (N,) — Z at anchor frame
    return rgb, queries, gt_z_anchor


def _median_at_query_pixels(depth_map: np.ndarray, queries_xy: np.ndarray, hw: tuple[int, int]) -> float:
    """Sample depth_map (Hd, Wd) at query (x, y) pixel coords specified in the
    original image space (H, W), then return median across valid samples."""
    H, W = hw
    Hd, Wd = depth_map.shape
    if H <= 0 or W <= 0 or queries_xy.size == 0:
        return float("nan")
    xs = np.clip((queries_xy[:, 0] / W * Wd).astype(np.int64), 0, Wd - 1)
    ys = np.clip((queries_xy[:, 1] / H * Hd).astype(np.int64), 0, Hd - 1)
    sampled = depth_map[ys, xs]
    valid = np.isfinite(sampled) & (sampled > 0)
    if not valid.any():
        return float("nan")
    return float(np.median(sampled[valid]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
    ap.add_argument("--model-name", default="da3metric-large")
    ap.add_argument("--clips-per-subset", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[da3-eval] device={device}, model={args.model_name}", flush=True)

    print("[da3-eval] loading DA3 ...", flush=True)
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model_name}").to(device)
    model.device = device  # the api class stashes this for ._prepare_model_inputs
    print(f"[da3-eval] loaded", flush=True)

    # Same stratified val sample as scale_est_v4/v5.
    train_clips, _ = official_train_test_split(
        args.data_root, subsets=["pstudio", "drivetrack", "adt"],
    )
    rng = random.Random(args.seed)
    by_sub: dict[str, list] = defaultdict(list)
    for c in train_clips:
        by_sub[_which_subset(Path(c))].append(c)
    val_clips: list[Path] = []
    for sub in ("pstudio", "drivetrack", "adt"):
        pool = by_sub.get(sub, [])
        if pool:
            val_clips.extend(rng.sample(pool, min(args.clips_per_subset, len(pool))))
    print(f"[da3-eval] {len(val_clips)} val clips ({args.clips_per_subset} per subset)", flush=True)

    per_sub_abs: dict[str, list[float]] = defaultdict(list)
    per_sub_rel: dict[str, list[float]] = defaultdict(list)

    for i, clip_path in enumerate(val_clips):
        sub = _which_subset(Path(clip_path))
        try:
            rgb, queries, gt_z_anchor = _load_anchor_frame_and_queries(Path(clip_path))
        except Exception as e:
            print(f"  [{i:2d}] {sub:<11}  {clip_path.name}: SKIP load ({type(e).__name__})")
            continue
        H, W = rgb.shape[:2]
        s_gt = float(np.median(gt_z_anchor[gt_z_anchor > 0])) if (gt_z_anchor > 0).any() else float("nan")
        if not np.isfinite(s_gt) or s_gt <= 0:
            print(f"  [{i:2d}] {sub:<11}  {clip_path.name}: SKIP no positive GT depth")
            continue
        try:
            pred = model.inference([rgb], process_res=504, export_format="mini_npz")
            depth = pred.depth[0]                # (Hd, Wd)
            s_da3 = _median_at_query_pixels(depth, queries[:, :2], (H, W))
        except Exception as e:
            print(f"  [{i:2d}] {sub:<11}  {clip_path.name}: SKIP DA3 ({type(e).__name__}: {e})")
            continue
        if not np.isfinite(s_da3) or s_da3 <= 0:
            print(f"  [{i:2d}] {sub:<11}  {clip_path.name}: SKIP no valid DA3 depth")
            continue
        abs_err = abs(s_da3 - s_gt)
        rel_err = abs_err / max(s_gt, 1e-3)
        per_sub_abs[sub].append(abs_err)
        per_sub_rel[sub].append(rel_err)
        print(f"  [{i:2d}] {sub:<11}  s_gt={s_gt:>7.3f}  s_da3={s_da3:>7.3f}  "
              f"abs_err={abs_err:>6.3f}m  rel_err={rel_err:>6.3f}", flush=True)

    print()
    print(f"{'subset':<12}  {'n':>4}  {'MAE [m]':>9}  {'rel_err':>9}")
    pooled_abs, pooled_rel = [], []
    for sub in sorted(per_sub_abs):
        n = len(per_sub_abs[sub])
        mae = sum(per_sub_abs[sub]) / n
        rel = sum(per_sub_rel[sub]) / n
        pooled_abs += per_sub_abs[sub]; pooled_rel += per_sub_rel[sub]
        print(f"{sub:<12}  {n:>4}  {mae:>9.3f}  {rel:>9.3f}")
    if pooled_abs:
        print(f"{'POOLED':<12}  {len(pooled_abs):>4}  "
              f"{sum(pooled_abs)/len(pooled_abs):>9.3f}  "
              f"{sum(pooled_rel)/len(pooled_rel):>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
