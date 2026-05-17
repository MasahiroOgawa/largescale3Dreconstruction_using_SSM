"""Render a few qualitative tracking MP4s from a trained Mamba-3 tracker.

Usage:
    uv run python scripts/render_tracker_video.py \\
        --ckpt outputs/runs/mamba3_tracker_v1/ckpt_30000.pt \\
        --out-dir outputs/eval_tracker/v1/viz \\
        --clips-per-subset 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from mamba3_tracker.data.tapvid3d import SUBSETS, list_clips, load_clip
from mamba3_tracker.model.tracker import Mamba3Tracker
from mamba3_tracker.viz.track_video import render_tracking_video


def _build_model(state: dict, device: torch.device) -> Mamba3Tracker:
    cfg = state["cfg"]
    model = Mamba3Tracker(
        dim=cfg["dim"], num_heads=cfg["num_heads"], state_dim=cfg["state_dim"],
        level_sizes=tuple(cfg["level_sizes"]),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def _predict_and_match(model, clip, device, amp_dtype, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    F = int(clip.images.shape[0]) if not max_frames else min(int(clip.images.shape[0]), max_frames)
    images = clip.images[:F].clone()
    H_orig, W_orig = images.shape[-2], images.shape[-1]
    img_size = model.image_size
    if H_orig != img_size or W_orig != img_size:
        images = torch.nn.functional.interpolate(
            images, size=(img_size, img_size),
            mode="bilinear", align_corners=False,
        )
    images = images.unsqueeze(0).to(device)

    sx = img_size / float(W_orig)
    sy = img_size / float(H_orig)
    N_q = clip.queries_xyt.shape[0]
    queries = clip.queries_xyt.clone()
    queries[:, 0] *= sx
    queries[:, 1] *= sy
    queries[:, 2] = queries[:, 2].clamp(max=F - 1)
    queries = queries.unsqueeze(0).to(device)
    qmask = torch.ones(1, N_q, dtype=torch.bool, device=device)

    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        pred = model(images, queries, qmask)
    delta = pred.xyz[0].float().cpu()
    pred_vis = torch.sigmoid(pred.vis_logits[0].float()).cpu()

    # Recover absolute predictions: p_t = p_query_gt + Δp_t.
    anchor_idx = clip.queries_xyt[:, 2].long().clamp(min=0, max=F - 1)
    gt_anchor_xyz = clip.tracks_XYZ[:F].gather(
        dim=0, index=anchor_idx.view(1, N_q, 1).expand(1, N_q, 3),
    ).squeeze(0)
    abs_pred = delta + gt_anchor_xyz.unsqueeze(0)
    return abs_pred.transpose(0, 1).numpy(), pred_vis.transpose(0, 1).numpy()


def _frames_to_uint8(images_01: torch.Tensor) -> np.ndarray:
    return (images_01.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--clips-per-subset", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--amp", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.amp]

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = _build_model(state, device)
    print(f"[viz] loaded {args.ckpt} (step={state.get('step', '?')})")

    for sub in args.subsets:
        clips = list_clips(args.data_root, [sub])[: args.clips_per_subset]
        print(f"[viz] {sub}: rendering {len(clips)} clips")
        for path in clips:
            try:
                clip = load_clip(path)
                tracks_NT3, vis_NT = _predict_and_match(model, clip, device, amp_dtype, args.max_frames)
                F = tracks_NT3.shape[1]
                frames = _frames_to_uint8(clip.images[:F])
                K = clip.K.numpy()
                out = args.out_dir / f"{sub}_{clip.clip_id}.mp4"
                render_tracking_video(frames, tracks_NT3, vis_NT, K, out, fps=args.fps)
                print(f"[viz] wrote {out}")
            except Exception as e:
                print(f"[viz] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}")
    print("[viz] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
