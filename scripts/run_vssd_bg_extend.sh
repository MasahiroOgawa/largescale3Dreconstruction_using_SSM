#!/usr/bin/env bash
# VSSD-beta,gamma: extend the 24500-step model while held-out loss is still falling.
#
# Continues from plateau_distill_vssd_bg/ckpt_4500.pt (= 20000 + 4500), the run that reached
# abs_rel 0.1098 -- the first continuation to improve on its own starting point.
#
# Asked for as +6000. At the measured 0.93 s/step that is 93 min of Phase-B, and with
# Phase-C (~16 min) and eval (~4 min) it would finish about 18:11, past the 18:00 deadline.
# So Phase-B is launched at 6000 and stopped by wall clock at CUTOFF instead: it trains as
# far as the clock allows, then the newest checkpoint goes through Phase-C and eval. A
# number always lands before the deadline, and no step of training is wasted deciding how
# many steps to ask for.
#
# --ckpt-every 1000 is what makes the cutoff safe: whatever the watchdog interrupts, there
# is a checkpoint at most 1000 steps behind it.
set -uo pipefail
cd "$(dirname "$0")/.."

CUTOFF=${CUTOFF:-$(date -d "today 17:20" +%s)}
prev=result/runs/plateau_distill_vssd_bg/ckpt_4500.pt
[ -f "$prev" ] || { echo "missing $prev"; exit 1; }

dist=result/runs/ext_distill_vssd_bg
ft=result/runs/ext_ft_vssd_bg
mkdir -p "$dist" "$ft"

echo "=== Phase-B +6000 from $prev, stop by $(date -d @"$CUTOFF" +%H:%M) ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --super 1 --sub 1 --variant vssd_bg --rope-all-layers \
    --init-ckpt "$prev" \
    --scheduler plateau --val-scenes eth3d:relief_2,eth3d:electro \
    --test-every 1000 --test-n-views 4 --test-n-batches 4 \
    --plateau-factor 0.5 --plateau-patience 2 --plateau-min-lr 1e-6 --plateau-metric L_D \
    --warmup-steps 100 --steps 6000 --ckpt-every 1000 --lr-attn 5.0e-5 \
    --out-dir "$dist" > "$dist/train.log" 2>&1 &
trainer=$!

while kill -0 "$trainer" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$CUTOFF" ]; then
    echo "=== cutoff reached, stopping Phase-B at $(grep -oE 'step +[0-9]+/6000' "$dist/train.log" | tail -1) ==="
    kill "$trainer" 2>/dev/null; sleep 20; kill -9 "$trainer" 2>/dev/null
    break
  fi
  sleep 30
done
wait "$trainer" 2>/dev/null

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
    --ckpt "$ft/ckpt_1000.pt" --label "VSSD-beta,gamma (24.5k extended)" \
    --out result/depth_eval_vssd_bg_ext.json 2>&1 | grep -E "abs_rel=" || true

echo "=== done ($(date -Is)) ==="
echo "  references: 9k 0.1153 | 20k 0.1244 | 24.5k 0.1098 | DA3-SMALL finetuned 0.0362"
