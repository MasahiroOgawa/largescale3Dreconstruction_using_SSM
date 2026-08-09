#!/usr/bin/env bash
# Continues run_norope_20k_all3.sh with the arm order changed to
# bidirectional -> VSSD-beta,gamma -> VSSD-gamma, so our own operator reports before 07:00
# rather than last.
#
# The original wrapper was killed rather than edited: bash reads a script by byte offset as
# it executes, so rewriting the trailing run_arm lines of a running script can make it
# resume mid-token. The bidirectional Phase-B it had already started is left untouched --
# this script waits on that pid, then runs its Phase-C and eval, then the remaining two arms
# in the new order.
#
# Recipe is unchanged from the original and still copies CM22/CM24: fresh Phase-B 20000 at
# 3e-4 with no 2-D RoPE anywhere, no --val-scenes, Phase-C 1000 on CM24's WSD shape.
set -uo pipefail
cd "$(dirname "$0")/.."

TRAINER=${TRAINER:?set TRAINER to the in-flight bidirectional Phase-B pid}
STEPS=${STEPS:-20000}

phase_c_and_eval () {
  local tag="$1" variant="$2" label="$3"
  local dist="result/runs/nr20k_distill_$tag" ft="result/runs/nr20k_ft_$tag"
  mkdir -p "$ft"
  local ck; ck=$(ls -t "$dist"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$ck" ] || { echo "[$tag] no checkpoint"; return 1; }

  echo "=== [$tag] Phase-C from $ck ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 3 --sub 3 --variant "$variant" --no-rope --init-ckpt "$ck" \
      --scheduler wsd --warmup-steps 100 --decay-steps 200 --steps 1000 \
      --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
      --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "[$tag] Phase-C FAILED"; return 1; }

  echo "=== [$tag] eval ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
      --ckpt "$ft/ckpt_1000.pt" --label "$label" \
      --out "result/depth_eval_nr20k_$tag.json" 2>&1 | grep -E "abs_rel=" || true
}

run_arm () {
  local tag="$1" variant="$2" label="$3"
  local dist="result/runs/nr20k_distill_$tag"
  mkdir -p "$dist"
  echo "=== [$tag] Phase-B $STEPS fresh, NO 2-D RoPE, lr 3e-4 ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 1 --sub 1 --variant "$variant" --no-rope \
      --scheduler wsd --warmup-steps 200 --decay-steps 500 \
      --steps "$STEPS" --ckpt-every 2000 --lr-attn 3.0e-4 \
      --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "[$tag] Phase-B FAILED"; return 1; }
  phase_c_and_eval "$tag" "$variant" "$label"
}

echo "=== waiting for the in-flight bidirectional Phase-B (pid $TRAINER) ($(date -Is)) ==="
while kill -0 "$TRAINER" 2>/dev/null; do sleep 60; done
echo "=== bidirectional Phase-B ended at $(grep -oE 'step +[0-9]+/20000' result/runs/nr20k_distill_bidir/train.log | tail -1) ($(date -Is)) ==="

phase_c_and_eval bidir   mamba3  "bidirectional (no 2-D RoPE, 20k)"
run_arm          vssd_bg vssd_bg "VSSD-beta,gamma (no 2-D RoPE, 20k)"
run_arm          vssd_gamma vssd "VSSD-gamma (no 2-D RoPE, 20k)"

echo "=== all arms done ($(date -Is)) ==="
grep -h abs_rel result/depth_eval_nr20k_*.json 2>/dev/null || true
echo "--- target: CM22 0.053 legacy ~= 0.042 on this evaluator ---"
echo "--- same arms WITH rope-all-layers, 9k: bidir 0.2036 | VSSD-g 0.1964 | VSSD-b,g 0.1153 ---"
echo "--- DA3-SMALL: zero-shot 0.0334, finetuned 0.0362 ---"
