"""Render 3D point trajectories in camera space (PNG, one per clip).

Each plot shows N sampled track trajectories as 3D polylines:
  * predicted track       : solid line, per-track colour
  * ground-truth track    : dashed line, same track colour, lower alpha
  * anchor frame position : marker dot

Same model + inference path as scripts/render_tracker_video.py — but instead
of compositing onto the input video, we draw the trajectories directly in
(X, Y, Z) camera coords using matplotlib's mplot3d.

Usage:
    uv run python scripts/render_3d_tracks.py \\
        --ckpt outputs/track_v18_<dt>/ckpt_30000.pt \\
        --out-dir outputs/track_v18_<dt>/viz_step30000 \\
        --subsets pstudio drivetrack \\
        --clips-per-subset 2 \\
        --max-frames 32 \\
        --max-tracks 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mamba3_tracker.data.tapvid3d import SUBSETS, list_clips, load_clip
from mamba3_tracker.model.tracker import Mamba3Tracker


def _build_model(state: dict, device: torch.device) -> Mamba3Tracker:
    cfg = state["cfg"]
    m = cfg.get("model", cfg)
    model = Mamba3Tracker(
        dim=int(m["dim"]), num_heads=int(m["num_heads"]),
        state_dim=int(m["state_dim"]),
        level_sizes=tuple(m.get("level_sizes", [32, 64])),
        num_iters=int(m.get("num_iters", 1)),
        use_correlation=bool(m.get("use_correlation", False)),
        encoder_kind=str(m.get("encoder_kind", "pyramid")),
        dinov2_model=str(m.get("dinov2_model", "facebook/dinov2-small")),
        dinov2_image_size=int(m.get("dinov2_image_size", 448)),
        dinov2_fuse_layers=m.get("dinov2_fuse_layers"),
        predict_scale=bool(m.get("predict_scale", False)),
    ).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def _predict_tracks(model, clip, device, amp_dtype, max_frames: int) -> np.ndarray:
    """Returns predicted absolute 3D positions in camera coords, shape (N_q, F, 3)."""
    F = int(clip.images.shape[0]) if not max_frames else min(int(clip.images.shape[0]), max_frames)
    images = clip.images[:F].clone()
    H_orig, W_orig = images.shape[-2], images.shape[-1]
    img_size = model.image_size
    if H_orig != img_size or W_orig != img_size:
        images = torch.nn.functional.interpolate(
            images, size=(img_size, img_size), mode="bilinear", align_corners=False,
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
    if pred.scale is not None:
        delta = float(pred.scale[0].float().cpu().item()) * delta
    a_n = clip.queries_xyt[:, 2].long().clamp(min=0, max=F - 1)
    init = clip.tracks_XYZ[a_n, torch.arange(N_q)]
    if F == 1:
        abs_pred = init.unsqueeze(0)
    else:
        delta_zero = delta.clone()
        delta_zero[0] = 0.0
        cs = delta_zero.cumsum(dim=0)
        cs_at_anchor = cs[a_n, torch.arange(N_q)]
        abs_pred = init.unsqueeze(0) + cs - cs_at_anchor.unsqueeze(0)
    return abs_pred.transpose(0, 1).numpy()                # (N_q, F, 3)


def _plot_clip_3d(
    pred_NF3: np.ndarray,         # (N_q, F, 3) predicted absolute 3D
    gt_NF3:   np.ndarray,         # (N_q, F, 3) GT absolute 3D
    vis_NF:   np.ndarray,         # (N_q, F)   bool: GT visibility per (n, t)
    anchor_n: np.ndarray,         # (N_q,)     anchor frame per track
    out_path: Path,
    title: str,
    max_tracks: int = 32,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    N_q, F, _ = pred_NF3.shape
    # Pick the longest-visible tracks first to ensure each plotted line has substance.
    vis_count = vis_NF.sum(axis=1)
    candidate = np.argsort(-vis_count)[: min(max_tracks * 2, N_q)]
    if len(candidate) > max_tracks:
        candidate = rng.choice(candidate, size=max_tracks, replace=False)
    sel = sorted(candidate.tolist())

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    for i, n in enumerate(sel):
        color = cmap(i % 20)
        m = vis_NF[n].astype(bool)
        if m.sum() < 2:
            continue
        # GT: dashed, low alpha
        ax.plot(gt_NF3[n, m, 0], gt_NF3[n, m, 1], gt_NF3[n, m, 2],
                linestyle="--", linewidth=1.0, color=color, alpha=0.55)
        # Predicted: solid
        ax.plot(pred_NF3[n, m, 0], pred_NF3[n, m, 1], pred_NF3[n, m, 2],
                linestyle="-",  linewidth=1.4, color=color, alpha=0.95)
        # Anchor marker
        a = int(anchor_n[n])
        if 0 <= a < F:
            ax.scatter(gt_NF3[n, a, 0], gt_NF3[n, a, 1], gt_NF3[n, a, 2],
                       s=18, color=color, edgecolors="black", linewidths=0.4)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m, depth)")
    ax.set_title(f"{title}\nsolid = predicted, dashed = GT, dot = anchor frame", fontsize=10)
    try:
        ax.set_box_aspect((1, 1, 1))                       # equal axis scaling
    except (AttributeError, ValueError):
        pass
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--clips-per-subset", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-tracks", type=int, default=32)
    ap.add_argument("--amp", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.amp]

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = _build_model(state, device)
    print(f"[3d] loaded {args.ckpt} (step={state.get('step', '?')})")

    for sub in args.subsets:
        clips = list_clips(args.data_root, [sub])[: args.clips_per_subset]
        print(f"[3d] {sub}: rendering {len(clips)} clips")
        for path in clips:
            try:
                clip = load_clip(path)
                pred = _predict_tracks(model, clip, device, amp_dtype, args.max_frames)
                F = pred.shape[1]
                gt = clip.tracks_XYZ[:F].numpy().transpose(1, 0, 2)         # (N, F, 3)
                vis = clip.visibility[:F].numpy().transpose(1, 0)            # (N, F)
                anchor = clip.queries_xyt[:, 2].long().clamp(0, F - 1).numpy()
                out = args.out_dir / f"{sub}_{clip.clip_id}_3d.png"
                _plot_clip_3d(
                    pred, gt, vis, anchor, out,
                    title=f"{sub}/{clip.clip_id}  (step {state.get('step', '?')})",
                    max_tracks=args.max_tracks,
                )
                print(f"[3d] wrote {out}")
            except Exception as e:
                print(f"[3d] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}")
    print("[3d] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
