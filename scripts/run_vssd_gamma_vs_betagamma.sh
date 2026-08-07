#!/usr/bin/env bash
# VSSD-gamma against VSSD-beta,gamma on ETH3D metric depth, at equal budget.
#
# The question is which of the two non-causal operators works as a drop-in mixer for
# DA3-SMALL's self-attention. Both arms get the identical recipe -- same teacher, same
# step count, same Phase-C -- so the comparison between them is clean even though the
# budget is below CM12's.
#
# TEACHER IS DA3-LARGE (--super 2), supervised on token-to-token structure rather than
# raw features. The elementwise feature loss needs equal channel counts, so 384 against
# LARGE's 1024 previously required a trainable Linear(384->1024) that was discarded before
# deployment -- the loophole that let CM30's student satisfy the objective from a ~40-dim
# subspace (last-layer rank 40 against CM12's 62, collapsed at every layer). A Gram matrix
# over tokens is (N,N) whichever channel count produced it, so there is no projector to
# exploit and the collapse is penalised directly.
#
# --rope-first-layer gives backbone block 0 DA3's own 2-D RoPE. DA3 starts RoPE at block 4,
# so blocks 0-3 hand the mixer no positional signal at all; softmax tolerates that, VSSD-gamma
# does not, since its mask collapses to a per-token vector with no |i-j| term of its own. The
# module is DA3's own instance, shared: zero parameters, zero buffers, total unchanged at
# 39.861M. Phase-B and Phase-C both take the flag so the student is built identically in each.
# The DA3-SMALL baseline does NOT get it -- it is compared exactly as published.
#
# STEPS = 12000, below CM12's 20000. Measured 1.648 s/step for LARGE + relational, so
# 20000 x 3 arms is ~30 h and does not fit the window; 12000 is the largest equal budget
# that lands before morning. These rows are therefore internally comparable but not
# directly comparable to the published CM24 number.
#
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-12000}
FT_STEPS=1000

run_arm() {                       # run_arm <variant> <tag>
  local variant="$1" tag="$2"
  local dist="result/runs/distill_${tag}" ft="result/runs/depth_ft_${tag}"
  mkdir -p "$dist" "$ft"

  echo "=== [$tag] Phase-B distill, $STEPS steps ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 2 --sub 1 --variant "$variant" --rope-first-layer \
      --steps "$STEPS" --ckpt-every 2000 \
      --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "[$tag] distill FAILED"; return 1; }

  # Newest checkpoint rather than a fixed name: if the arm was cut short, Phase-C should
  # still run on whatever it reached instead of failing on a missing file.
  local init; init=$(ls -t "$dist"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$init" ] || { echo "[$tag] no distill checkpoint"; return 1; }
  echo "=== [$tag] Phase-C depth fine-tune from $init ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 3 --sub 3 --variant "$variant" --rope-first-layer --init-ckpt "$init" \
      --steps "$FT_STEPS" --warmup-steps 100 --decay-steps 200 \
      --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
      --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "[$tag] Phase-C FAILED"; return 1; }
  echo "=== [$tag] done ($(date -Is)) ==="
}

eval_ckpt() {                     # eval_ckpt <dir> <tag>
  local d="$1" tag="$2"
  local ck; ck=$(ls -t "$d"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$ck" ] || { echo "[$tag] nothing to evaluate"; return 0; }
  echo "=== [$tag] ETH3D terrains eval on $ck ($(date -Is)) ==="
  # Timeout so one hang cannot eat the others' slot -- last night's baseline eval died in
  # the CPU-side fusion stage after the forward passes had already succeeded.
  timeout 3600 env CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_da3_bench_eval.py \
      model.path="$ck" eval.datasets=[eth3d] eval.modes=[recon_posed] \
      workspace.work_dir="$d/eval" 2>&1 | tee "$d/eval.log" \
      || echo "[$tag] eval did not complete -- checkpoint kept at $ck"
}

# Priority order: if the window runs out, it runs out on the arm we can already cite.
run_arm vssd_bg vssd_bg           # ours -- the one that must land
run_arm vssd    vssd_gamma        # the operator the tracker deploys
run_arm mamba3  bidir             # bidirectional, for a same-budget three-way

eval_ckpt result/runs/depth_ft_vssd_bg      vssd_bg
eval_ckpt result/runs/depth_ft_vssd_gamma   vssd_gamma
eval_ckpt result/runs/depth_ft_bidir        bidir
eval_ckpt result/runs/depth_ft_da3small_baseline da3small   # last night's, eval never finished

echo "=== all arms complete ($(date -Is)) ==="
