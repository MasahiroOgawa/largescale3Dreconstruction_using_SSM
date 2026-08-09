#!/usr/bin/env bash
# All three operators with NO 2-D RoPE anywhere, Phase-B 20000 + Phase-C 1000.
#
# This is an attempt to reproduce CM22/CM24's 0.053 and then read the other two operators
# against it, so it copies that recipe rather than the recipes we have been using since:
#
#   Fresh Phase-B, not warm-started. CM12 trained the mixer from scratch for 20000 steps,
#     and every checkpoint we could warm-start from carries 2-D RoPE in 8-12 blocks -- the
#     very thing being removed.
#   lr_attn 3e-4 in Phase-B. That is CM12's rate for a randomly-initialised mixer. The 5e-5
#     used lately is a continuation rate and would under-train a fresh one.
#   No --val-scenes. It holds 2 of the 10 ETH3D scenes out of training, and CM22 trained on
#     all 10. Reproduction beats instrumentation here; the plateau schedule is dropped for
#     the same reason.
#   Phase-C: 1000 steps, lr_attn 1e-5 / lr_head 1e-5 / lr_other 3e-5, WSD warmup 100 decay
#     200 -- CM24's shape, which is what produced 0.0513.
#
# Bidirectional runs FIRST: it is the only variant with a legacy number to reproduce, so it
# is the one that decides whether 2-D RoPE explains the 0.204-vs-0.053 gap. The other two
# are only interpretable once it lands.
#
# Expected if 2-D RoPE was the cause: roughly 0.042 on our evaluator, NOT 0.053. Our
# evaluator scores the same zero-shot DA3-SMALL checkpoint at 0.0334 where the 2026-04-23
# one scored 0.0417, so a faithful reproduction of 0.053 reads about 20% lower here.
# Anything near 0.20 means 2-D RoPE was not the cause and the mamba-ssm kernel change is
# the remaining suspect.
#
# ~5 h per arm at the measured 0.9 s/step; bidirectional lands ~00:30, VSSD-gamma ~05:40,
# VSSD-beta,gamma ~11:00. Only the first two are done by 07:00.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-20000}

run_arm () {
  local tag="$1" variant="$2" label="$3"
  local dist="result/runs/nr20k_distill_$tag" ft="result/runs/nr20k_ft_$tag"
  mkdir -p "$dist" "$ft"

  echo "=== [$tag] Phase-B $STEPS fresh, NO 2-D RoPE, lr 3e-4 ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 1 --sub 1 --variant "$variant" --no-rope \
      --scheduler wsd --warmup-steps 200 --decay-steps 500 \
      --steps "$STEPS" --ckpt-every 2000 --lr-attn 3.0e-4 \
      --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "[$tag] Phase-B FAILED"; return 1; }

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

run_arm bidir      mamba3   "bidirectional (no 2-D RoPE, 20k)"
run_arm vssd_gamma vssd     "VSSD-gamma (no 2-D RoPE, 20k)"
run_arm vssd_bg    vssd_bg  "VSSD-beta,gamma (no 2-D RoPE, 20k)"

echo "=== all arms done ($(date -Is)) ==="
grep -h abs_rel result/depth_eval_nr20k_*.json 2>/dev/null || true
echo "--- target: CM22 0.053 legacy ~= 0.042 on this evaluator ---"
echo "--- same arms WITH rope-all-layers, 9k: bidir 0.2036 | VSSD-g 0.1964 | VSSD-b,g 0.1153 ---"
echo "--- DA3-SMALL: zero-shot 0.0334, finetuned 0.0362 ---"
