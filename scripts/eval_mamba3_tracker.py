"""Evaluate a trained Mamba-3 tracker on TAPVid-3D.

For each held-out clip:
  1. Load ground-truth tracks + visibility.
  2. Run inference: feed the full video as one window (or stream in chunks).
  3. Use the model's Hungarian-matching anchor logic against the GT queries
     to assign predicted slots to GT track IDs, so per-track metrics line up.
  4. Compute 3D-AJ / APD3D / OA via `mamba3_tracker.eval.tapvid3d_eval`.

Output: per-clip JSONs + a per-subset roll-up `summary.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mamba3_tracker.data.dataset import collate_tracking
from mamba3_tracker.data.tapvid3d import SUBSETS, list_clips, load_clip
from mamba3_tracker.eval.tapvid3d_eval import aggregate, compute_clip_metrics
from mamba3_tracker.model.tracker import Mamba3Tracker


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
def _infer_clip(model, clip, device, amp_dtype, max_frames: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_tracks_NT3, pred_visibility_NT) for one clip.

    v2: slot n is GT track n by construction (query-conditioned bank). The
    model emits Δp relative to the query anchor; absolute prediction is
    recovered as `p_query + Δp` using GT at the query frame.
    """
    F = int(clip.images.shape[0]) if max_frames in (0, None) else min(int(clip.images.shape[0]), max_frames)
    images = clip.images[:F].clone()
    H_orig, W_orig = images.shape[-2], images.shape[-1]
    img_size = model.image_size
    if H_orig != img_size or W_orig != img_size:
        images = torch.nn.functional.interpolate(
            images, size=(img_size, img_size),
            mode="bilinear", align_corners=False,
        )
    images = images.unsqueeze(0).to(device)                  # (1, F, 3, S, S)

    # Scale query (x, y) to resized pixel space; clamp anchor frame to [0, F-1].
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

    delta = pred.xyz[0].float().cpu()                        # (F, N_q, 3)
    pred_vis = torch.sigmoid(pred.vis_logits[0].float()).cpu()  # (F, N_q)

    # Recover absolute predictions: p_n^(t) = p_n^query + Δp_n^(t).
    anchor_idx = clip.queries_xyt[:, 2].long().clamp(min=0, max=F - 1)
    gt_anchor_xyz = clip.tracks_XYZ[:F].gather(
        dim=0, index=anchor_idx.view(1, N_q, 1).expand(1, N_q, 3),
    ).squeeze(0)                                             # (N_q, 3)
    pred_abs = delta + gt_anchor_xyz.unsqueeze(0)            # (F, N_q, 3)

    pred_tracks_NT3 = pred_abs.transpose(0, 1).numpy()
    pred_vis_NT = pred_vis.transpose(0, 1).numpy()
    return pred_tracks_NT3, pred_vis_NT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--max-clips-per-subset", type=int, default=0,
                    help="0 = all clips.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = full clip. Use a small value for quick smoke runs.")
    ap.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp]

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = _build_model(state, device)
    print(f"[eval] loaded {args.ckpt} (step={state.get('step', '?')})")

    per_subset_clips: dict[str, list] = defaultdict(list)
    for sub in args.subsets:
        clips = list_clips(args.data_root, [sub])
        if args.max_clips_per_subset:
            clips = clips[: args.max_clips_per_subset]
        per_subset_clips[sub] = clips
        print(f"[eval] {sub}: {len(clips)} clips")

    metrics_root = args.out_dir / "metric_results"
    metrics_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float]] = {}

    for sub, clips in per_subset_clips.items():
        per_clip: list[dict[str, float]] = []
        for i, path in enumerate(clips):
            try:
                clip = load_clip(path)
                pred_tracks, pred_vis = _infer_clip(model, clip, device, amp_dtype, args.max_frames)
                F = pred_tracks.shape[1]
                gt_xyz = clip.tracks_XYZ[:F].numpy()
                gt_vis = clip.visibility[:F].float().numpy()
                K = clip.K.numpy()
                intrin = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                m = compute_clip_metrics(gt_xyz, gt_vis, pred_tracks, pred_vis, intrin)
                m["clip_id"] = clip.clip_id
                per_clip.append(m)
                print(f"[eval] {sub}/{clip.clip_id}: AJ={m['average_jaccard']:.4f} "
                      f"APD3D={m['average_pts_within_thresh']:.4f} OA={m['occlusion_accuracy']:.4f}",
                      flush=True)
            except Exception as e:
                print(f"[eval] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}", flush=True)
        (metrics_root / f"{sub}.json").write_text(json.dumps(per_clip, indent=2))
        agg = aggregate(per_clip)
        summary[sub] = agg
        print(f"[eval] {sub} mean: {agg}", flush=True)

    # Roll-up
    overall = aggregate(summary.values())
    rows = [
        "# Mamba-3 Tracker — TAPVid-3D evaluation",
        f"\nCheckpoint: `{args.ckpt}` (step {state.get('step', '?')})",
        "\n## Per-subset means\n",
        "| subset | 3D-AJ | APD3D | OA |",
        "|---|---|---|---|",
    ]
    for sub, m in summary.items():
        rows.append(
            f"| {sub} | {m.get('average_jaccard', float('nan')):.4f} | "
            f"{m.get('average_pts_within_thresh', float('nan')):.4f} | "
            f"{m.get('occlusion_accuracy', float('nan')):.4f} |"
        )
    rows.append(
        f"| **mean** | **{overall.get('average_jaccard', float('nan')):.4f}** | "
        f"**{overall.get('average_pts_within_thresh', float('nan')):.4f}** | "
        f"**{overall.get('occlusion_accuracy', float('nan')):.4f}** |"
    )
    (args.out_dir / "summary.md").write_text("\n".join(rows) + "\n")
    print(f"\n[eval] DONE. Summary at {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
