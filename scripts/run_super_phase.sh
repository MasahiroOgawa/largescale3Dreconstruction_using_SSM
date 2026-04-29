#!/usr/bin/env bash
# Run one super-phase: sub-1 (attn) → sub-2 (head) → sub-3 (all) → sub-4 (eval) → plot.
#
# Usage:
#   bash scripts/run_super_phase.sh <super> [init_ckpt]
#     <super>     1, 2, or 3
#     [init_ckpt] starting ckpt for sub-1; default = DA3-SMALL pretrained
#
# Output dirs: outputs/runs/sp{super}_sub{1,2,3}/, sp{super}_loss.png, sp{super}_eval.log
set -e

SUPER=$1
INIT=${2:-}
STEPS=500

cd "$(dirname "$0")/.."

OUT="outputs/runs/sp${SUPER}_sub1"
LOG="outputs/runs/sp${SUPER}_sub1.log"
echo "=== sub ${SUPER}-1 (attn-only) ==="
INIT_FLAG=""
[ -n "$INIT" ] && INIT_FLAG="--init-ckpt $INIT"
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -u -m ssm3d.train.train_super \
  --super $SUPER --sub 1 --steps $STEPS --ckpt-every 250 \
  $INIT_FLAG --out-dir $OUT > $LOG 2>&1
echo "=== sub ${SUPER}-1 done. tail: ==="
tail -3 outputs/runs/sp${SUPER}_sub1/log.txt

OUT="outputs/runs/sp${SUPER}_sub2"
LOG="outputs/runs/sp${SUPER}_sub2.log"
echo "=== sub ${SUPER}-2 (head adapt) ==="
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -u -m ssm3d.train.train_super \
  --super $SUPER --sub 2 --steps $STEPS --ckpt-every 250 \
  --init-ckpt outputs/runs/sp${SUPER}_sub1/ckpt_${STEPS}.pt \
  --out-dir $OUT > $LOG 2>&1
echo "=== sub ${SUPER}-2 done. tail: ==="
tail -3 outputs/runs/sp${SUPER}_sub2/log.txt

OUT="outputs/runs/sp${SUPER}_sub3"
LOG="outputs/runs/sp${SUPER}_sub3.log"
echo "=== sub ${SUPER}-3 (full unfreeze) ==="
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -u -m ssm3d.train.train_super \
  --super $SUPER --sub 3 --steps $STEPS --ckpt-every 250 \
  --init-ckpt outputs/runs/sp${SUPER}_sub2/ckpt_${STEPS}.pt \
  --out-dir $OUT > $LOG 2>&1
echo "=== sub ${SUPER}-3 done. tail: ==="
tail -3 outputs/runs/sp${SUPER}_sub3/log.txt

echo "=== sub ${SUPER}-4 (eval) ==="
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -u -m ssm3d.eval.phase4_evaluator \
  --ckpt outputs/runs/sp${SUPER}_sub3/ckpt_${STEPS}.pt \
  --max-images 12 --image-size 504 \
  --pose-only \
  > outputs/runs/sp${SUPER}_eval.log 2>&1
grep -A 12 "STUDENT" outputs/runs/sp${SUPER}_eval.log

echo "=== plotting super ${SUPER} loss curves ==="
uv run python scripts/plot_super_phase_loss.py \
  --super $SUPER \
  --logs outputs/runs/sp${SUPER}_sub1/log.txt \
         outputs/runs/sp${SUPER}_sub2/log.txt \
         outputs/runs/sp${SUPER}_sub3/log.txt \
  --out outputs/runs/sp${SUPER}_loss.png

echo "=== super-phase ${SUPER} COMPLETE ==="
