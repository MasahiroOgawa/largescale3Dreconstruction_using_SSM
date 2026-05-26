#!/usr/bin/env bash
# Render all "latest" results for a tracker training run in one shot:
#   * tracking MP4s from a checkpoint (per subset/clip), and
#   * training/validation loss curves + motion-ratio plots from the run's history.
#
# MP4s go in <run_dir>/viz_step<N>/, plots go in <run_dir>/plots/
# (see memory/feedback_output_dir_naming.md — datetime only at run-dir level).
#
# Usage:
#   scripts/render_all_latest_results.sh                                 # latest run dir, latest ckpt
#   scripts/render_all_latest_results.sh outputs/track_v17_<dt>          # specific run dir, latest ckpt
#   scripts/render_all_latest_results.sh outputs/track_v17_<dt>/ckpt_5000.pt   # specific ckpt
#   USE_CPU=0 scripts/render_all_latest_results.sh             # use GPU (only when training not running)
#   CLIPS_PER_SUBSET=4 scripts/render_all_latest_results.sh    # more clips
#   MAX_FRAMES=64 scripts/render_all_latest_results.sh         # longer windows
#
# Argument resolution:
#   no arg                 -> ls -td outputs/track_v*_*/ | head -1 ; latest ckpt
#   path is a directory    -> that dir's latest ckpt
#   path is a .pt file     -> that exact ckpt
#
# Outputs:
#   <run_dir>/viz_step<N>/*.mp4         per-clip rendered tracking videos
#   <run_dir>/viz_step<N>/render.log    stdout/stderr of this run
#   <run_dir>/plots/training_curve.png  per-term train+val loss curves
#   <run_dir>/plots/motion_ratio.png    per-subset motion ratio vs step

set -euo pipefail

ARG="${1:-}"

if [ -z "$ARG" ]; then
    RUN_DIR="$(ls -td outputs/track_v*_*/ 2>/dev/null | head -1)"
    RUN_DIR="${RUN_DIR%/}"
    if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
        echo "error: no run dir found under outputs/track_v*_*/" >&2
        exit 1
    fi
    CKPT=$(ls -t "$RUN_DIR"/ckpt_*.pt 2>/dev/null | head -1)
elif [ -d "$ARG" ]; then
    RUN_DIR="${ARG%/}"
    CKPT=$(ls -t "$RUN_DIR"/ckpt_*.pt 2>/dev/null | head -1)
elif [ -f "$ARG" ]; then
    CKPT="$ARG"
    RUN_DIR="$(dirname "$CKPT")"
else
    echo "error: $ARG is neither a directory nor a file" >&2
    exit 1
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "error: no checkpoint found (RUN_DIR=$RUN_DIR)" >&2
    exit 1
fi
STEP=$(basename "$CKPT" .pt | sed 's/ckpt_//')
OUT="${RUN_DIR}/viz_step${STEP}"
mkdir -p "$OUT"
LATEST_CKPT="$CKPT"

USE_CPU="${USE_CPU:-1}"
CLIPS_PER_SUBSET="${CLIPS_PER_SUBSET:-2}"
MAX_FRAMES="${MAX_FRAMES:-32}"
SUBSETS="${SUBSETS:-pstudio drivetrack}"
SPLIT="${SPLIT:-all}"          # all | minival | full_eval — for eval-time viz use minival

echo "run    : $RUN_DIR"
echo "ckpt   : $LATEST_CKPT (step $STEP)"
echo "output : $OUT"
echo "device : $([ "$USE_CPU" = "1" ] && echo CPU || echo GPU)"
echo "subsets: $SUBSETS  | clips/subset: $CLIPS_PER_SUBSET  | max-frames: $MAX_FRAMES  | split: $SPLIT"
echo

CMD=(
    uv run python scripts/render_tracker_video.py
    --ckpt "$LATEST_CKPT"
    --out-dir "$OUT"
    --subsets $SUBSETS
    --clips-per-subset "$CLIPS_PER_SUBSET"
    --max-frames "$MAX_FRAMES"
    --split "$SPLIT"
    --fps 15
)
if [ "$USE_CPU" = "1" ]; then
    CUDA_VISIBLE_DEVICES="" "${CMD[@]}" --amp fp32 2>&1 | tee "$OUT/render.log"
else
    "${CMD[@]}" --amp bf16 2>&1 | tee "$OUT/render.log"
fi

# 3D-space trajectory plots (PNG, one per clip). Same model + clip set
# as the 2D MP4 step above; uses matplotlib mplot3d so no extra deps.
echo
echo "[3d] rendering 3D-space track plots"
CMD_3D=(
    uv run python scripts/render_3d_tracks.py
    --ckpt "$LATEST_CKPT"
    --out-dir "$OUT"
    --subsets $SUBSETS
    --clips-per-subset "$CLIPS_PER_SUBSET"
    --max-frames "$MAX_FRAMES"
    --split "$SPLIT"
)
if [ "$USE_CPU" = "1" ]; then
    CUDA_VISIBLE_DEVICES="" "${CMD_3D[@]}" --amp fp32 2>&1 | tee -a "$OUT/render.log"
else
    "${CMD_3D[@]}" --amp bf16 2>&1 | tee -a "$OUT/render.log"
fi

# Space-time diagrams (3 planar projections of (x,y,z) world coords
# against frame index). Same clip set, same model.
echo
echo "[st] rendering space-time track plots (xy / yz / zx × time)"
CMD_ST=(
    uv run python scripts/render_space_time_tracks.py
    --ckpt "$LATEST_CKPT"
    --out-dir "$OUT"
    --subsets $SUBSETS
    --clips-per-subset "$CLIPS_PER_SUBSET"
    --max-frames "$MAX_FRAMES"
    --split "$SPLIT"
)
if [ "$USE_CPU" = "1" ]; then
    CUDA_VISIBLE_DEVICES="" "${CMD_ST[@]}" --amp fp32 2>&1 | tee -a "$OUT/render.log"
else
    "${CMD_ST[@]}" --amp bf16 2>&1 | tee -a "$OUT/render.log"
fi

echo
echo "[plot] rendering training-curve + motion-ratio plots"
uv run python scripts/plot_training_curves.py --run-dir "$RUN_DIR" 2>&1 | tee -a "$OUT/render.log"

# Final summary — absolute paths so users can copy-paste and find them
# even after the surrounding logs scroll out of view.
RUN_ABS="$(realpath "$RUN_DIR")"
OUT_ABS="$(realpath "$OUT")"
PLOTS_ABS="$(realpath "$RUN_DIR/plots")"

echo
echo "================================================================"
echo "DONE — outputs for step $STEP"
echo "================================================================"
echo "run dir : $RUN_ABS"
echo
echo "MP4s    ($OUT_ABS):"
for f in "$OUT"/*.mp4; do
    [ -f "$f" ] && echo "  $(realpath "$f")"
done
echo
echo "3D PNGs ($OUT_ABS):"
for f in "$OUT"/*_3d.png; do
    [ -f "$f" ] && echo "  $(realpath "$f")"
done
echo
echo "3D HTML ($OUT_ABS) — open in a browser for interactive rotate/zoom:"
for f in "$OUT"/*_3d.html; do
    [ -f "$f" ] && echo "  $(realpath "$f")"
done
echo
echo "Space-time PNGs ($OUT_ABS) — xy / yz / zx vs frame index:"
for f in "$OUT"/*_st.png; do
    [ -f "$f" ] && echo "  $(realpath "$f")"
done
echo
echo "Space-time HTML ($OUT_ABS) — interactive 1×3 subplots:"
for f in "$OUT"/*_st.html; do
    [ -f "$f" ] && echo "  $(realpath "$f")"
done
echo
echo "plots   ($PLOTS_ABS):"
for f in "$PLOTS_ABS"/*.png; do
    [ -f "$f" ] && echo "  $f"
done
echo
echo "log     : $OUT_ABS/render.log"
echo "================================================================"
