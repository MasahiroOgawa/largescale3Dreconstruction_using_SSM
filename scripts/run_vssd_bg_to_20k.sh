#!/usr/bin/env bash
# VSSD-beta,gamma alone: take the DA3-SMALL teacher distillation to 20000 steps.
#
# The paper needs this operator to beat DA3-SMALL; the other two arms are stopped so the
# whole GPU budget goes here. Totals, counting only SMALL-teacher steps:
#
#   4000  already done   abs_rel 0.1561  (worse than the 11000 LARGE-teacher run's 0.1344)
#  +5000  stage 1        evaluated; stop here if it beats DA3-SMALL's 0.0362
# +11000  stage 2        only if it does not -- 4k+5k+11k = 20k SMALL-teacher steps
#
# Stage 2 is conditional and decided from the measured number, not assumed: if stage 1
# already wins there is no reason to spend three more hours, and if it loses the run
# continues without waiting for someone to notice.
#
# The 11000 DA3-LARGE steps underneath are not counted in that 20k. CM24 reached 0.0513
# with 20000 pure SMALL from a DA3-SMALL init, so even the full run here is a different
# recipe, and falling short of 0.0513 would not by itself convict the operator.
set -uo pipefail
cd "$(dirname "$0")/.."

DA3_SMALL_ABS_REL=0.0362          # the number to beat, measured today, same protocol

run_stage() {                      # run_stage <steps> <init-ckpt> <tag>
  local steps="$1" prev="$2" tag="$3"
  local dist="result/runs/distill_vssd_bg_$tag" ft="result/runs/depth_ft_vssd_bg_$tag"
  mkdir -p "$dist" "$ft"

  echo "=== [$tag] Phase-B +$steps from $prev ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 1 --sub 1 --variant vssd_bg --rope-all-layers \
      --init-ckpt "$prev" --steps "$steps" --ckpt-every 1000 \
      --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "[$tag] Phase-B FAILED"; return 1; }

  local init; init=$(ls -t "$dist"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$init" ] || { echo "[$tag] no checkpoint"; return 1; }

  echo "=== [$tag] Phase-C from $init ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 3 --sub 3 --variant vssd_bg --rope-all-layers --init-ckpt "$init" \
      --steps 1000 --warmup-steps 100 --decay-steps 200 \
      --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
      --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "[$tag] Phase-C FAILED"; return 1; }

  echo "=== [$tag] eval ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
      --ckpt "$ft/ckpt_1000.pt" --label "VSSD-beta,gamma ($tag)" \
      --out "result/depth_eval_vssd_bg_$tag.json" 2>&1 | grep -E "abs_rel=" || true
}

beats_baseline() {                 # beats_baseline <json>
  python3 -c "
import json, sys
try:
    v = json.load(open('$1'))['abs_rel']
except Exception:
    sys.exit(1)          # unreadable result is not a win
sys.exit(0 if v < $DA3_SMALL_ABS_REL else 1)"
}

run_stage 5000 result/runs/distill_vssd_bg_small/ckpt_4000.pt s1 || exit 1

if beats_baseline result/depth_eval_vssd_bg_s1.json; then
  echo "=== stage 1 beats DA3-SMALL ($DA3_SMALL_ABS_REL); stopping ($(date -Is)) ==="
  exit 0
fi

echo "=== stage 1 did not beat DA3-SMALL; continuing to 20k ($(date -Is)) ==="
run_stage 11000 "$(ls -t result/runs/distill_vssd_bg_s1/ckpt_*.pt | head -1)" s2 || exit 1

echo "=== done ($(date -Is)) ==="
grep -h abs_rel result/depth_eval_vssd_bg_*.json 2>/dev/null || true
