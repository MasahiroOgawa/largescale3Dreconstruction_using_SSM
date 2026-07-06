# largescale3Dreconstruction_using_SSM

3D point tracking from monocular video using a lightweight SSM refiner on top of optical flow + metric depth.

**Current best (v35):** metric-AJ 0.234, 7.7 fps on drivetrack subset.

---

## How it works

```
video frames
    └─► SEA-RAFT optical flow  ─► 2D tracks (uv, visibility)
    └─► Depth-Anything-3 depth ─► per-frame depth maps (cached)
                                           │
                            Mamba3V35Refiner (0.44 M trainable params)
                            ┌─ frozen DINOv3-ViT-S/16 visual features
                            ├─ SSM depth refinement  → Δlog_z
                            └─ 2D correction         → Δuv
                                           │
                                   3D tracks (XYZ) in camera space
```

DA3 depth is precomputed once offline (~10 fps) and cached at
`~/data/tapvid3d_da3/<subset>/<clip>.npz`.  All per-frame inference
(SEA-RAFT + v35 refinement) runs at ~7.7 fps.

Evaluated on [TAPVid-3D](https://tapvid3d.github.io/) (drivetrack / pstudio / adt subsets).

---

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd largescale3Dreconstruction_using_SSM
uv sync
```

The venv is managed by **`uv`** — `pip` is never used.  
`uv sync` installs all pinned deps including the local submodule at
`third_party/depth-anything-3`.

Requirements: CUDA GPU (bf16 training/inference), Python 3.11.

---

## Viewing v35 visual results

Best checkpoint: `result/v35_20260701/ckpt_20000.pt`

### TAPVid-style (filled circles + fading line trails)

```bash
uv run python scripts/render_tracks.py \
    --method v35 \
    --ckpt result/v35_20260701/ckpt_20000.pt \
    --out-dir result/$(date +%Y%m%d-%H%M)_viz_v35_tapvid \
    --style tapvid \
    --subsets drivetrack \
    --clips-per-subset 2 \
    --split minival
```

### D4RT-style (vivid rainbow dot trails)

Inspired by [D4RT (DeepMind, 2024)](https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/):
each track gets a distinct vivid hue; the current frame is a bright dot with
a white sparkle; the tail is alpha-fading dots that shrink toward the past.

```bash
uv run python scripts/render_tracks.py \
    --method v35 \
    --ckpt result/v35_20260701/ckpt_20000.pt \
    --out-dir result/$(date +%Y%m%d-%H%M)_viz_v35_d4rt \
    --style d4rt \
    --subsets drivetrack \
    --clips-per-subset 2 \
    --split minival
```

Per clip the script writes:
- `<subset>_<clip_id>.mp4` — 2D tracking video (TAPVid or D4RT style)
- `<subset>_<clip_id>_3d.png` — static 3D trajectory (pred solid / GT dashed)
- `<subset>_<clip_id>_3d.html` — interactive 3D trajectory (plotly, open in browser)
- `<subset>_<clip_id>_st.png` — space-time plot (X/Y/Z vs time)
- `metrics.json` — real-metric 3D error (metres) per clip

`--scaling none|median|anchor` controls depth scaling applied to the plots
(default `none` = raw output; `median` = global scale that TAPVid-3D AJ uses).

---

## Running evaluation

```bash
# Minival eval on all 3 subsets
uv run python scripts/eval_metric3d.py \
    --method v35 \
    --ckpt result/v35_20260701/ckpt_20000.pt \
    --split minival \
    --out-dir result/$(date +%Y%m%d-%H%M)_metric3d_v35
```

---

## Key results (minival, 50 clips per subset)

| Method | norm-AJ | abs-AJ | metric-AJ | fps |
|--------|---------|--------|-----------|-----|
| SEA-RAFT + DA3 (baseline) | 0.083 | — | 0.147 | ~10 |
| v35 (ours) | — | — | 0.234 | 7.7 |

Full tables with TAPIP3D and SpatialTrackerV2 comparisons are in
`doc/vmamba3_3dpointtrack/vmamba3_3dpointtrack.tex`.

---

## Project structure

```
src/mamba3_tracker/
    model/depth_refined_tracker.py   Mamba3V35Refiner (main model)
    data/tapvid3d.py                 TAPVid-3D data loader
    eval/tapvid3d_official_metrics.py  3D-AJ / APD3D evaluation
    viz/track_video.py               render_tracking_video, render_tracking_video_d4rt

scripts/
    eval_metric3d.py                 full minival evaluation
    render_tracks.py                 qualitative rendering (--method v35 --style d4rt)
    train_depth_refined_tracker.py   v35 training

third_party/
    depth-anything-3/                DA3 submodule (read-only)
    TrackCraft3R/                    TrackCraft3R submodule (for comparison)

result/
    v35_20260701/ckpt_20000.pt       best v35 checkpoint
    YYYYMMDD-HHMM_<name>/            eval + viz outputs (datetime-first naming)
```
