#!/usr/bin/env bash
# Phase-C + eval for the in-flight Phase-B of run_vssd_bg_extend.sh.
#
# That script wrapped Phase-B in a 17:20 wall-clock watchdog, sized for a hard 18:00
# deadline. The deadline turned out to be internal, so the watchdog was killed and Phase-B
# left running to its full 6000 steps. Killing the wrapper also removed the Phase-C and eval
# stages that followed it, which is what this script restores.
#
# It waits on the training PID rather than polling for a checkpoint file: a checkpoint
# appears every 1000 steps, so "newest checkpoint exists" would fire long before training
# ends. Waiting on the process is the only signal that Phase-B is actually done.
set -uo pipefail
cd "$(dirname "$0")/.."

TRAINER=${TRAINER:?set TRAINER to the Phase-B pid}
dist=result/runs/ext_distill_vssd_bg
ft=result/runs/ext_ft_vssd_bg
mkdir -p "$ft"

echo "=== waiting for Phase-B (pid $TRAINER) ($(date -Is)) ==="
while kill -0 "$TRAINER" 2>/dev/null; do sleep 30; done
echo "=== Phase-B ended at $(grep -oE 'step +[0-9]+/6000' "$dist/train.log" | tail -1) ($(date -Is)) ==="

init=$(ls -t "$dist"/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$init" ] || { echo "no checkpoint written; nothing to evaluate"; exit 1; }

echo "=== Phase-C from $init ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 3 --sub 3 --variant vssd_bg --rope-all-layers --init-ckpt "$init" \
    --scheduler plateau --val-scenes eth3d:relief_2,eth3d:electro \
    --test-every 250 --test-n-views 4 --test-n-batches 4 \
    --plateau-factor 0.5 --plateau-patience 2 --plateau-min-lr 1e-7 --plateau-metric L_D \
    --warmup-steps 50 --steps 1000 \
    --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
    --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "Phase-C FAILED"; exit 1; }

echo "=== eval ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
    --ckpt "$ft/ckpt_1000.pt" --label "VSSD-beta,gamma (24.5k + 6k extended)" \
    --out result/depth_eval_vssd_bg_ext.json 2>&1 | grep -E "abs_rel=" || true

echo "=== done ($(date -Is)) ==="
echo "  references: 9k 0.1153 | 20k 0.1244 | 24.5k 0.1098 | DA3-SMALL finetuned 0.0362"
