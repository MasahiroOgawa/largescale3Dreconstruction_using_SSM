#!/usr/bin/env bash
# Symmetric ETH3D baseline: DA3-SMALL given the SAME adaptation as our mixer.
#
# Paper Table 2 compares a Vision-Mamba-3 backbone that was distilled *and* had its
# DualDPT head fine-tuned on ETH3D against a DA3-SMALL evaluated zero-shot. That is not a
# like-for-like comparison, and it favours us -- our one winning metric (delta<1.25) comes
# from 1000 steps of ETH3D training the baseline never got. This run removes that
# confound by putting un-patched DA3-SMALL through the identical depth fine-tune.
#
# Recipe = CM24 (doc/PLAN.md:1281, the repo's retained pipeline, which supersedes the CM22
# numbers the paper currently quotes): 1000 steps, WSD shape (warmup 100 / decay 200),
# lr_attn 1e-5 / lr_head 1e-5 / lr_other 3e-5, DualDPT unfrozen, terrains held out.
#
# LR-group mapping, from train_super.py:352-357 -- `head` is dpt+cam_dec and `other` is
# everything else including the bridge, so CM24's lr_dpt=1e-5 becomes --lr-head and its
# lr_bridge=3e-5 becomes --lr-other. Getting this backwards would silently produce an
# unfair baseline, which is worse than no baseline.
#
# --no-mamba3-swap leaves DA3's own softmax attention in place; those weights then land in
# the `attn` group and train at the same 1e-5 our mixer does. That is the point: same
# budget, same schedule, same data, same held-out scene, only the operator differs.
#
# Waits for the CIFAR T=1025 chain to exit first -- the GPU must be exclusive or the
# latency numbers that chain is still producing get corrupted.
set -euo pipefail
cd "$(dirname "$0")/.."

WAIT_PID="${1:-}"
OUT=result/runs/depth_ft_da3small_baseline

if [ -n "$WAIT_PID" ]; then
  echo "[baseline] waiting for PID $WAIT_PID (CIFAR chain) to finish ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[baseline] PID $WAIT_PID gone at $(date -Is); GPU should be free"
  sleep 30
fi

mkdir -p "$OUT"
echo "=== DA3-SMALL symmetric depth fine-tune ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
    --no-mamba3-swap \
    --steps 1000 --warmup-steps 100 --decay-steps 200 \
    --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
    --out-dir "$OUT" 2>&1 | tee "$OUT/train.log"

# Eval is attempted here so the run is self-contained, but the checkpoint is the durable
# artifact: if these hydra keys need adjusting, re-run just this step against ckpt_1000.pt
# rather than repeating the fine-tune.
echo "=== ETH3D terrains eval ($(date -Is)) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_da3_bench_eval.py \
    model.path="$OUT/ckpt_1000.pt" \
    eval.datasets=[eth3d] \
    eval.modes=[recon_posed] \
    workspace.work_dir="$OUT/eval" 2>&1 | tee "$OUT/eval.log" || {
      echo "[baseline] eval step failed -- checkpoint is at $OUT/ckpt_1000.pt, fix the"
      echo "           evaluator args and re-run only the eval."; exit 1; }

echo "=== baseline complete ($(date -Is)) ==="
