#!/usr/bin/env bash
# End-to-end eval pipeline: efficiency bench + DA3 official benchmark on
# baseline DA3-SMALL and the final VSSD ckpt, then plot.
#
# Run from project root AFTER `outputs/runs/vssd_da3_v1/ckpt_*.pt` exists.
#
# All DA3-evaluator invocations run inside a memory-capped systemd-run scope
# (per memory/feedback_da3_bench_num_fusion_workers.md). Without the cap a
# single TSDF fusion can reach 28 GB RSS, trigger global OOM, and SIGKILL
# the parent tmux session.

set -euo pipefail
EVAL_ROOT="outputs/eval_vssd_full"
CKPT_DIR="outputs/runs/vssd_da3_v1"

# pick the largest-step ckpt
CKPT=$(ls -1 "$CKPT_DIR"/ckpt_*.pt | sed 's/.*ckpt_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | awk '{print $2}')
echo "[pipeline] using ckpt: $CKPT"

mkdir -p "$EVAL_ROOT/accuracy" "$EVAL_ROOT/efficiency"

# Run DA3 evaluator inside a 22 GB memory-capped scope so OOMs stay local.
# Requires `loginctl enable-linger $USER` to be set so user@1000.service
# survives across sessions; bench survival is bounded by the cap, not the
# parent tmux scope.
DA3_EVAL_RUN() {
    systemd-run --user --scope --quiet \
        -p MemoryMax=22G -p MemorySwapMax=8G -p OOMPolicy=continue \
        -- "$@"
}

# ── 1. Efficiency benchmark ─────────────────────────────────────────────
echo "[pipeline] efficiency benchmark"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python scripts/bench_efficiency_patched.py \
        --variants da3_small vssd \
        --sizes 224 392 504 672 \
        --n-views 4 8 12 \
        --out-dir "$EVAL_ROOT/efficiency"

# ── 2. DA3 official benchmark — baseline DA3-SMALL ──────────────────────
echo "[pipeline] DA3 evaluator: baseline DA3-SMALL"
DA3_EVAL_RUN bash -c "
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python scripts/run_da3_bench_eval.py \
        model.path=depth-anything/DA3-SMALL \
        eval.datasets='[eth3d,7scenes,scannetpp,hiroom,dtu]' \
        eval.modes='[recon_posed,pose]' \
        inference.num_fusion_workers=1 \
        workspace.work_dir=$EVAL_ROOT/accuracy/da3_small
"

# ── 3. DA3 official benchmark — VSSD-patched ckpt ──────────────────────
echo "[pipeline] DA3 evaluator: VSSD-DA3"
DA3_EVAL_RUN bash -c "
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python scripts/run_da3_bench_eval.py \
        model.path=$CKPT \
        eval.datasets='[eth3d,7scenes,scannetpp,hiroom,dtu]' \
        eval.modes='[recon_posed,pose]' \
        inference.num_fusion_workers=1 \
        workspace.work_dir=$EVAL_ROOT/accuracy/vssd_da3_v1
"

# ── 4. Plot ─────────────────────────────────────────────────────────────
echo "[pipeline] plotting comparison"
uv run python scripts/plot_da3_vs_vssd.py \
    --eval-root "$EVAL_ROOT" \
    --models da3_small vssd_da3_v1 \
    --datasets eth3d 7scenes scannetpp hiroom dtu

echo "[pipeline] done — see $EVAL_ROOT/comparison.{png,pdf} + summary.md"
