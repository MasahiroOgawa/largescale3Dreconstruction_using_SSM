#!/usr/bin/env bash
# VSSD-beta,gamma continuation at a learning rate suited to a warm start.
#
# Every previous continuation ran Phase B at 3e-4, because scope="attn" hard-coded it and
# ignored --lr-attn. That is CM12's rate for a freshly initialised mixer; applied to an
# already-converged one it re-heats the solution, and the feature loss oscillates instead of
# descending (0.264 -> 0.352 -> 0.280 -> 0.335 -> 0.359 within one stage). Depth degraded
# monotonically with more of it: 0.1344 -> 0.1561 -> 0.2105.
#
# 1e-5 here, matching what Phase C uses for the same parameters, so the two stages no longer
# disagree by 30x about how fast this mixer should move. Warm-started from the best checkpoint
# available rather than the most-degraded one: distill_vssd_bg/ckpt_11000.pt is the DA3-LARGE
# result that scored 0.1344, before any of the over-heated SMALL-teacher steps.
set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "run_vssd_bg[_]to_20k" > /dev/null; do sleep 60; done
sleep 20

prev=result/runs/distill_vssd_bg/ckpt_11000.pt
dist=result/runs/distill_vssd_bg_lowlr
ft=result/runs/depth_ft_vssd_bg_lowlr
mkdir -p "$dist" "$ft"

echo "=== Phase-B 6000 steps at lr 1e-5, from $prev ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 1 --sub 1 --variant vssd_bg --rope-all-layers \
    --init-ckpt "$prev" --steps 6000 --ckpt-every 2000 --lr-attn 1.0e-5 \
    --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "Phase-B FAILED"; exit 1; }

echo "=== Phase-C ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 3 --sub 3 --variant vssd_bg --rope-all-layers \
    --init-ckpt "$(ls -t $dist/ckpt_*.pt | head -1)" \
    --steps 1000 --warmup-steps 100 --decay-steps 200 \
    --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
    --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "Phase-C FAILED"; exit 1; }

echo "=== eval ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
    --ckpt "$ft/ckpt_1000.pt" --label "VSSD-beta,gamma (lr 1e-5)" \
    --out result/depth_eval_vssd_bg_lowlr.json 2>&1 | grep -E "abs_rel=" || true
echo "=== done ($(date -Is)) ==="
