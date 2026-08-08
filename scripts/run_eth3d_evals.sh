#!/usr/bin/env bash
# ETH3D terrains eval for the four trained arms, at the protocol that does not OOM.
#
# The overnight chain trained all three arms successfully and then died in the first eval,
# at `eth3d fusion: 0%`, taking the whole process group with it -- a system OOM kill, which
# is uncatchable, which is why the timeout and || fallbacks never fired. Two causes, both
# fixed here rather than worked around:
#
#   eval.max_frames=12          the default 100 let terrains fuse 42 views; the published
#                               CM numbers used a 12-view eval, so this both fits in RAM
#                               and matches the protocol those numbers were measured under.
#   inference.num_fusion_workers=1   the default 4 holds four TSDF volumes at once on a
#                               31 GB host. Serial fusion is slower and bounded.
#
# Each arm runs as its own process, and the driver is setsid'd, so if one still dies it
# cannot take the others with it -- the failure mode that cost us all four evals.
set -uo pipefail
cd "$(dirname "$0")/.."

ARMS=(
  "vssd_bg:result/runs/depth_ft_vssd_bg/ckpt_1000.pt"
  "vssd_gamma:result/runs/depth_ft_vssd_gamma/ckpt_1000.pt"
  "bidir:result/runs/depth_ft_bidir/ckpt_1000.pt"
  "da3small:result/runs/depth_ft_da3small_baseline/ckpt_1000.pt"
)

for entry in "${ARMS[@]}"; do
  tag="${entry%%:*}"; ckpt="${entry#*:}"
  out="result/eval_${tag}"
  if [ ! -f "$ckpt" ]; then echo "[$tag] missing $ckpt -- skipped"; continue; fi
  mkdir -p "$out"
  echo "=== [$tag] eval on $ckpt ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_da3_bench_eval.py \
      model.path="$ckpt" \
      eval.datasets=[eth3d] \
      eval.scenes=[terrains] \
      eval.modes=[recon_posed] \
      eval.max_frames=12 \
      inference.num_fusion_workers=1 \
      workspace.work_dir="$out" > "$out/eval.log" 2>&1
  rc=$?
  # Peak RSS is worth knowing even on success: if an arm ran close to the limit, the next
  # protocol change should account for it rather than rediscover the OOM.
  echo "=== [$tag] rc=$rc ($(date -Is)) ==="
  grep -iE "f-score|fscore|chamfer|precision|recall|auc" "$out/eval.log" | tail -6
done

echo "=== all evals done ($(date -Is)) ==="
