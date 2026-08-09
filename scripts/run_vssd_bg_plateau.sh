#!/usr/bin/env bash
# VSSD-beta,gamma continued from the 20000-step checkpoint under a closed-loop LR schedule.
#
# Why this run exists. Every previous continuation picked an LR shape up front -- 3e-4 flat,
# then cosine to zero -- and could not notice when it stopped helping. The 9k -> 20k run is
# the clearest case: per-scene feature loss was already flat across 20 of 22 scenes by step
# 4000, and it still spent 7000 more steps at a rate that was buying nothing, finishing
# WORSE than where it started (abs_rel 0.1153 -> 0.1244). An open-loop schedule cannot react
# to that; ReduceLROnPlateau can, which is the whole point of using it here.
#
# The validation signal is real held-out loss, not training loss relabelled. --val-scenes
# removes those scenes from the training pool as well as measuring on them; the trainer's
# older --test-scenes only measured, so its "TEST loss" line was computed on scenes the
# model was simultaneously training on. relief_2 and electro are held out for this. The eval
# scene, terrains, is rejected by the parser -- tuning the LR on the scene we report would
# leak it.
#
# LR starts at 5e-5, half the 1e-4 that failed. The mixer has already seen 20000 steps, so
# this is a refinement, not a restart; plateau will take it down further on its own if that
# is still too high, and the floor is 1e-6.
set -uo pipefail
cd "$(dirname "$0")/.."

prev=result/runs/w20k_distill_vssd_bg/ckpt_11000.pt   # = 20000 cumulative
[ -f "$prev" ] || { echo "missing $prev"; exit 1; }

dist=result/runs/plateau_distill_vssd_bg
ft=result/runs/plateau_ft_vssd_bg
mkdir -p "$dist" "$ft"

STEPS=${STEPS:-6000}

echo "=== Phase-B +${STEPS} from $prev, plateau on held-out loss ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 1 --sub 1 --variant vssd_bg --rope-all-layers \
    --init-ckpt "$prev" \
    --scheduler plateau --val-scenes eth3d:relief_2,eth3d:electro \
    --test-every 1000 --test-n-views 4 --test-n-batches 4 \
    --plateau-factor 0.5 --plateau-patience 2 --plateau-min-lr 1e-6 \
    --warmup-steps 100 --steps "$STEPS" --ckpt-every 1000 --lr-attn 5.0e-5 \
    --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "Phase-B FAILED"; exit 1; }

echo "=== Phase-C ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 3 --sub 3 --variant vssd_bg --rope-all-layers \
    --init-ckpt "$(ls -t $dist/ckpt_*.pt | head -1)" \
    --scheduler plateau --val-scenes eth3d:relief_2,eth3d:electro \
    --test-every 250 --test-n-views 4 --test-n-batches 4 \
    --plateau-factor 0.5 --plateau-patience 2 --plateau-min-lr 1e-7 \
    --warmup-steps 50 --steps 1000 \
    --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
    --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "Phase-C FAILED"; exit 1; }

echo "=== eval ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
    --ckpt "$ft/ckpt_1000.pt" --label "VSSD-beta,gamma (20k + plateau)" \
    --out result/depth_eval_vssd_bg_plateau.json 2>&1 | grep -E "abs_rel=" || true

echo "=== done ($(date -Is)) ==="
echo "  references: 9k 0.1153 | 20k 0.1244 | DA3-SMALL finetuned 0.0362"
