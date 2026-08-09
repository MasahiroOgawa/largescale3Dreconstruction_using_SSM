#!/usr/bin/env bash
# Reproduce the 0.053 bidirectional result, with abs_rel measured every 5000 Phase-B steps.
#
# Staged rather than one 20000-step run followed by four evals, because the point is to see
# whether abs_rel actually falls with Phase-B length. Staging puts the first real number at
# ~08:50 instead of ~12:35, and if abs_rel rises instead of falling there is no reason to
# spend the remaining hours -- the trajectory is the result, not just the endpoint.
#
# abs_rel needs Phase-C: Phase-B trains the mixer against the teacher with the depth head
# frozen, so a Phase-B checkpoint has no meaningful depth output. Each stage therefore runs
# a full Phase-C 1000 from that stage's checkpoint and evaluates that.
#
# RoPE: the DEFAULT setting, which is the one configuration never yet tried. The swap
# inherits DA3's own 2-D RoPE in blocks 4-11 and leaves 0-3 bare, so this is "swap the mixer
# and change nothing else" -- the most faithful reading of the original experiment. The two
# extremes are already measured and neither reproduces:
#     12/12 blocks (--rope-all-layers), 9k   -> 0.2036
#      0/12 blocks (--no-rope),        20k   -> 0.3019
#
# LR is a constant 3e-4, CM12's rate for a fresh mixer. Constant rather than per-stage WSD:
# a decay-and-rewarm at every 5000-step boundary would be a different schedule from the
# single 20000-step run this is meant to imitate, and re-heating a settled mixer is the
# failure that produced 0.1344 -> 0.1561 -> 0.2105 earlier. The Phase-C at each stage keeps
# CM24's WSD shape, which is what produced 0.0513.
#
# ~95 min per stage: evals at roughly 08:50, 10:25, 12:00, 13:35.
set -uo pipefail
cd "$(dirname "$0")/.."

VARIANT=${VARIANT:-mamba3}
SEG=${SEG:-5000}
STAGES=${STAGES:-4}

prev=""
for i in $(seq 1 "$STAGES"); do
  total=$(( i * SEG ))
  dist="result/runs/rp_distill_bidir_${total}"
  ft="result/runs/rp_ft_bidir_${total}"
  mkdir -p "$dist" "$ft"

  echo "=== [stage $i] Phase-B ${SEG} steps -> ${total} cumulative ($(date -Is)) ==="
  init_arg=()
  [ -n "$prev" ] && init_arg=(--init-ckpt "$prev")
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 1 --sub 1 --variant "$VARIANT" \
      "${init_arg[@]}" \
      --scheduler wsd --warmup-steps 50 --decay-steps 0 \
      --steps "$SEG" --ckpt-every "$SEG" --lr-attn 3.0e-4 \
      --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "[stage $i] Phase-B FAILED"; exit 1; }

  prev=$(ls -t "$dist"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$prev" ] || { echo "[stage $i] no checkpoint"; exit 1; }

  echo "=== [stage $i] Phase-C from $prev ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 3 --sub 3 --variant "$VARIANT" --init-ckpt "$prev" \
      --scheduler wsd --warmup-steps 100 --decay-steps 200 --steps 1000 \
      --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
      --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "[stage $i] Phase-C FAILED"; exit 1; }

  echo "=== [stage $i] eval at ${total} Phase-B steps ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
      --ckpt "$ft/ckpt_1000.pt" --label "bidirectional repro, Phase-B ${total}" \
      --out "result/depth_eval_rp_bidir_${total}.json" 2>&1 | grep -E "abs_rel=" || true

  echo "--- trajectory so far ---"
  for f in result/depth_eval_rp_bidir_*.json; do
    [ -f "$f" ] && python3 -c "
import json;d=json.load(open('$f'));print(f\"    {d['label']:38s} abs_rel={d['abs_rel']:.4f}\")"
  done
done

echo "=== done ($(date -Is)) ==="
echo "  target: CM22 0.053 legacy (~0.042 on this evaluator)"
echo "  already measured: rope-all 9k 0.2036 | no-rope 20k 0.3019 | DA3-SMALL 0.0334 zero-shot"
