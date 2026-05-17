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
from mamba3_tracker.train.loss import _hungarian_match_anchor
from mamba3_tracker.viz.track_video import render_tracking_video


def _build_model(state: dict, device: torch.device) -> Mamba3Tracker:
    cfg = state["cfg"]
    model = Mamba3Tracker(
        dim=cfg["dim"], num_heads=cfg["num_heads"], state_dim=cfg["state_dim"],
        num_tracks=cfg["num_tracks"], level_sizes=tuple(cfg["level_sizes"]),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def _predict_and_match(model, clip, device, amp_dtype, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    F = int(clip.images.shape[0]) if not max_frames else min(int(clip.images.shape[0]), max_frames)
    images = clip.images[:F].unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        pred = model(images)
    pred_xyz = pred.xyz[0].float().cpu()
    pred_vis = torch.sigmoid(pred.vis_logits[0].float()).cpu()

    N_q = clip.queries_xyt.shape[0]
    anchor_idx = clip.queries_xyt[:, 2].long().clamp(min=0, max=F - 1)
    gt_anchor_xyz = torch.gather(
        clip.tracks_XYZ[:F], dim=0,
        index=anchor_idx.view(1, N_q, 1).expand(1, N_q, 3),
    ).squeeze(0)
    pred_anchor = pred_xyz[0]
    assign = _hungarian_match_anchor(
        pred_anchor.unsqueeze(0), gt_anchor_xyz.unsqueeze(0),
        torch.ones(1, N_q, dtype=torch.bool),
    )[0]
    matched_xyz = pred_xyz[:, assign, :]
    matched_vis = pred_vis[:, assign]
    return matched_xyz.transpose(0, 1).numpy(), matched_vis.transpose(0, 1).numpy()


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
