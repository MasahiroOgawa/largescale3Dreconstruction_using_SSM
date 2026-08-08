#!/usr/bin/env bash
# Continue all three operators' Phase B against the DA3-SMALL teacher, warm-started from
# the DA3-LARGE checkpoints, then re-run Phase C and evaluate.
#
# The DA3-LARGE arms landed at abs_rel 0.1344 / 0.1536 / 0.1623 against DA3-SMALL's 0.0362,
# so the accuracy half of the objective is not met. Every SMALL-teacher run in the record
# beat every LARGE-teacher one (CM24 0.0513, CM12 0.0676, against CM30 0.0943 and CM20
# 0.1739), and switching the teacher is the single change most likely to close it.
#
# Warm start rather than from scratch, per instruction: the mixer already has 11000 steps
# of structure and 5 h does not buy 11000 fresh ones. Worth knowing the risk -- those steps
# pulled it toward DA3-LARGE's feature space, so the starting point may sit in a basin the
# SMALL teacher has to climb out of before it improves. The intermediate checkpoints below
# will show which, since a warm start that is hurting shows up as the first few thousand
# steps going sideways.
#
# Teacher is --super 1, which restores the dim-matched L2+cosine feature loss; the
# relational Gram loss exists only because DA3-LARGE's 1024-d features cannot be subtracted
# from the student's 384-d. Nothing else changes: same operators, same --rope-all-layers,
# same Phase-C recipe, so teacher is the only variable against the runs it is compared to.
#
# Budget: 4000 Phase-B steps per arm at ~1.0-1.16 s/step, plus ~16 min Phase C and ~3 min
# eval each, is ~4.5 h of the 5 h available. Fresh output directories so the LARGE results
# survive as the comparison.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-4000}
FT_STEPS=1000

for entry in "vssd_bg:VSSD-beta,gamma" "vssd:VSSD-gamma" "mamba3:bidirectional"; do
  variant="${entry%%:*}"; label="${entry#*:}"
  case "$variant" in
    vssd_bg) tag=vssd_bg ;;
    vssd)    tag=vssd_gamma ;;
    mamba3)  tag=bidir ;;
  esac
  prev="result/runs/distill_${tag}/ckpt_11000.pt"
  dist="result/runs/distill_${tag}_small"
  ft="result/runs/depth_ft_${tag}_small"

  if [ ! -f "$prev" ]; then echo "[$tag] missing $prev -- skipped"; continue; fi
  mkdir -p "$dist" "$ft"

  echo "=== [$tag] Phase-B continue, SMALL teacher, $STEPS steps from $prev ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 1 --sub 1 --variant "$variant" --rope-all-layers \
      --init-ckpt "$prev" --steps "$STEPS" --ckpt-every 1000 \
      --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "[$tag] Phase-B FAILED"; continue; }

  init=$(ls -t "$dist"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$init" ] || { echo "[$tag] no Phase-B checkpoint"; continue; }

  echo "=== [$tag] Phase-C from $init ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 3 --sub 3 --variant "$variant" --rope-all-layers --init-ckpt "$init" \
      --steps "$FT_STEPS" --warmup-steps 100 --decay-steps 200 \
      --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
      --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "[$tag] Phase-C FAILED"; continue; }

  echo "=== [$tag] eval ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
      --ckpt "$ft/ckpt_${FT_STEPS}.pt" --label "$label (SMALL teacher)" \
      --out "result/depth_eval_${tag}_small.json" 2>&1 | grep -E "abs_rel=" || true
done

echo "=== all arms complete ($(date -Is)) ==="
echo "--- DA3-LARGE arms, for comparison ---"
grep -h "abs_rel" result/depth_eval_*.json 2>/dev/null || true
