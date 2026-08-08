#!/usr/bin/env bash
# VSSD-beta,gamma: 5000 more SMALL-teacher Phase-B steps, taking the total to 20000
# (11000 DA3-LARGE + 9000 DA3-SMALL), then Phase C and the same eval protocol.
#
# The first 4000 SMALL-teacher steps made it worse, 0.1344 -> 0.1561, while the feature
# loss against DA3-SMALL fell 0.4214 -> 0.3304. Features moving out of DA3-LARGE's space
# but not yet into DA3-SMALL's, with only 1000 Phase-C steps to re-fit against a target
# still in motion. This tests whether finishing the transition recovers it.
#
# Waits for the running three-arm chain so the two do not share the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "run_small[_]teacher_continue" > /dev/null; do sleep 60; done
sleep 30

prev=result/runs/distill_vssd_bg_small/ckpt_4000.pt
dist=result/runs/distill_vssd_bg_small2
ft=result/runs/depth_ft_vssd_bg_small2
mkdir -p "$dist" "$ft"

echo "=== [vssd_bg] Phase-B +5000 from $prev ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 1 --sub 1 --variant vssd_bg --rope-all-layers \
    --init-ckpt "$prev" --steps 5000 --ckpt-every 1000 \
    --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "Phase-B FAILED"; exit 1; }

init=$(ls -t "$dist"/ckpt_*.pt | head -1)
echo "=== [vssd_bg] Phase-C from $init ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 3 --sub 3 --variant vssd_bg --rope-all-layers --init-ckpt "$init" \
    --steps 1000 --warmup-steps 100 --decay-steps 200 \
    --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
    --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "Phase-C FAILED"; exit 1; }

echo "=== [vssd_bg] eval ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
    --ckpt "$ft/ckpt_1000.pt" --label "VSSD-beta,gamma (20000 total)" \
    --out result/depth_eval_vssd_bg_20k.json 2>&1 | grep -E "abs_rel=" || true
echo "=== done ($(date -Is)) ==="
