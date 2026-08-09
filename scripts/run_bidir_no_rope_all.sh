#!/usr/bin/env bash
# The ablation that should have been run before any of the overnight arms.
#
# Question: why does the current bidirectional variant reach abs_rel 0.204 where CM22/CM24
# reached 0.053? Five things differ; four are now accounted for.
#
#   Phase-B length (9000 vs CM12's 20000) -- ruled out. VSSD-beta,gamma at 9k/20k/24.5k/30.5k
#     gives 0.1153 / 0.1244 / 0.1098 / 0.1120, a band with no trend, and the +6000 extension
#     degraded held-out loss monotonically through an LR cut. More Phase-B steps do not help.
#   Evaluator -- measured, not guessed: ours scores the same zero-shot DA3-SMALL checkpoint at
#     0.0334 where the 2026-04-23 one scored 0.0417, ~25% apart. Real, but a fraction of a 4x gap.
#   Phase-C recipe -- identical. Checked against PLAN.md rows 21/22: 1000 steps, lr_attn 1e-5,
#     lr_dpt 1e-5, lr_other 3e-5, augmentation on, image_size 504.
#   mamba-ssm kernel -- changed, and not something this run can isolate.
#
# That leaves --rope-all-layers, which every overnight arm used and CM22 did not have at all
# (doc/attention 10.4: "produced without 2-D RoPE"). It is the only difference that changes
# what every pretrained block computes: DA3-SMALL's weights never saw a 2-D rotation, and we
# apply one to B and C in all 12 blocks, so every pretrained MLP downstream receives features
# in a basis it was not trained on.
#
# This run is the 0.204 arm with that flag removed and nothing else touched -- same variant,
# same 9000 steps, same cosine schedule, same Phase-C. One variable.
#
#   rope-all-layers ON  = 0.2036  (result/depth_eval_on_bidir.json)
#   rope-all-layers OFF = this run
set -uo pipefail
cd "$(dirname "$0")/.."

dist=result/runs/norope_distill_bidir
ft=result/runs/norope_ft_bidir
mkdir -p "$dist" "$ft"

STEPS=${STEPS:-9000}

echo "=== Phase-B $STEPS, bidirectional, NO --rope-all-layers ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 1 --sub 1 --variant mamba3 \
    --scheduler cosine --warmup-steps 200 --steps "$STEPS" --ckpt-every 2000 \
    --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "Phase-B FAILED"; exit 1; }

init=$(ls -t "$dist"/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$init" ] || { echo "no checkpoint"; exit 1; }

echo "=== Phase-C from $init ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 3 --sub 3 --variant mamba3 --init-ckpt "$init" \
    --scheduler cosine --warmup-steps 100 --steps 1000 \
    --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
    --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "Phase-C FAILED"; exit 1; }

echo "=== eval ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
    --ckpt "$ft/ckpt_1000.pt" --label "bidirectional, no rope-all-layers (9000)" \
    --out result/depth_eval_norope_bidir.json 2>&1 | grep -E "abs_rel=" || true

echo "=== done ($(date -Is)) ==="
echo "  same arm WITH --rope-all-layers: 0.2036 | CM22 legacy: 0.053 | DA3-SMALL: 0.0362"
