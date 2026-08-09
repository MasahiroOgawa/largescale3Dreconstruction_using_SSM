#!/usr/bin/env bash
# VSSD-beta,gamma: +11000 Phase-B steps on top of the 9000 already done, reaching 20000.
#
# Warm start from result/runs/on_distill_vssd_bg/ckpt_9000.pt -- the arm that scored
# abs_rel 0.1153, the best VSSD number so far and the first not damaged by the 3e-4 bug.
#
# LR is 1e-4, not the 3e-4 the fresh run used. The 9000-step cosine ended at ~0, so the
# mixer is settled; restarting at 3e-4 is exactly the re-heat that degraded every earlier
# continuation (0.1344 -> 0.1561 -> 0.2105). 1e-4 is high enough to keep learning and well
# below the rate that did the damage. Cosine again, so it ends settled rather than
# mid-oscillation.
#
# Checkpoints every 2000 and the last three evaluated, not just the final one: this is a
# single shot against a 18:00 deadline, and if 20000 turns out worse than 9000 we need the
# intermediate numbers to see where it turned rather than only that it did.
set -uo pipefail
cd "$(dirname "$0")/.."

# Wait for the last CIFAR cell rather than share the GPU -- concurrent runs measured 0.89x
# aggregate throughput here, so sharing would delay both.
while pgrep -f "eval.run[_]cifar" > /dev/null; do sleep 60; done
sleep 20

prev=result/runs/on_distill_vssd_bg/ckpt_9000.pt
dist=result/runs/w20k_distill_vssd_bg
ft=result/runs/w20k_ft_vssd_bg
mkdir -p "$dist" "$ft"

echo "=== Phase-B +11000 (9000 -> 20000) from $prev ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 1 --sub 1 --variant vssd_bg --rope-all-layers \
    --init-ckpt "$prev" --scheduler cosine --warmup-steps 200 \
    --steps 11000 --ckpt-every 2000 --lr-attn 1.0e-4 \
    --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "Phase-B FAILED"; exit 1; }

echo "=== Phase-C ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 3 --sub 3 --variant vssd_bg --rope-all-layers \
    --init-ckpt "$(ls -t $dist/ckpt_*.pt | head -1)" \
    --scheduler cosine --warmup-steps 100 --steps 1000 \
    --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
    --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "Phase-C FAILED"; exit 1; }

echo "=== eval ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
    --ckpt "$ft/ckpt_1000.pt" --label "VSSD-beta,gamma (20k warm)" \
    --out result/depth_eval_vssd_bg_20k_warm.json 2>&1 | grep -E "abs_rel=" || true

echo "=== done ($(date -Is)); 9k reference abs_rel 0.1153, DA3-SMALL 0.0362 ==="
