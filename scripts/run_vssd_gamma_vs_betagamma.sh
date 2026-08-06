#!/usr/bin/env bash
# VSSD-gamma against VSSD-beta,gamma on ETH3D metric depth, at equal budget.
#
# The question is which of the two non-causal operators works as a drop-in mixer for
# DA3-SMALL's self-attention. Both arms get the identical recipe -- same teacher, same
# step count, same Phase-C -- so the comparison between them is clean even though the
# budget is below CM12's.
#
# TEACHER IS DA3-SMALL (--super 1), not DA3-LARGE, and that is forced rather than chosen.
# train_super.py:505 gates feature distillation on `super_phase == 1`, the DistillProjector
# that CM20 used to bridge 384->1024 no longer exists in src/, and FEAT_LAYERS = (5,7,9,11)
# are the 12-block SMALL teacher's indices with no LARGE equivalent. Running --super 2 today
# would therefore distill with no feature term at all -- strictly weaker than CM20, which
# already collapsed to 0.1739. Restoring the LARGE path is development work, not a run.
# The teacher choice does not bias gamma against beta,gamma: both arms share it.
#
# STEPS = 12000, not CM12's 20000, to fit both arms plus Phase-C and eval before the
# morning. Checkpoints every 2000 mean that if an arm runs long there is always a usable
# earlier checkpoint, and the trend across them shows whether accuracy had plateaued --
# which is the thing to check before reading anything into the final number.
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
      --super 1 --sub 1 --variant "$variant" \
      --steps "$STEPS" --ckpt-every 2000 \
      --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "[$tag] distill FAILED"; return 1; }

  # Newest checkpoint rather than a fixed name: if the arm was cut short, Phase-C should
  # still run on whatever it reached instead of failing on a missing file.
  local init; init=$(ls -t "$dist"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$init" ] || { echo "[$tag] no distill checkpoint"; return 1; }
  echo "=== [$tag] Phase-C depth fine-tune from $init ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 3 --sub 3 --variant "$variant" --init-ckpt "$init" \
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

run_arm vssd_bg vssd_bg           # ours first: it is the one that must land
run_arm vssd    vssd_gamma

eval_ckpt result/runs/depth_ft_vssd_bg      vssd_bg
eval_ckpt result/runs/depth_ft_vssd_gamma   vssd_gamma
eval_ckpt result/runs/depth_ft_da3small_baseline da3small   # last night's, eval never finished

echo "=== all arms complete ($(date -Is)) ==="
