#!/usr/bin/env bash
# Overnight: all three operators with 2-D RoPE on the FIRST BLOCK ONLY.
#
# What "first layer only" means here, since it is not obvious. DA3-SMALL carries its own 2-D
# RoPE in blocks 4-11 and none in blocks 0-3. --rope-all-layers fills 0-3;
# --rope-first-layer fills block 0 and leaves 1-3 bare. Neither touches 4-11 -- removing
# DA3's own RoPE would damage pretrained weights rather than ablate an addition of ours, so
# this is "add it to the first block only", not "have it nowhere else".
#
# Why: every arm so far used --rope-all-layers, and CM22 -- the run that reached 0.053 --
# had no 2-D RoPE at all. That flag is the one difference between the two that changes what
# every pretrained block computes. This sits between the two extremes.
#
# Warm-started from each variant's latest Phase-B checkpoint, as asked, which also means the
# three arms do not start from equal training budgets:
#   VSSD-beta,gamma  ext_distill_vssd_bg/ckpt_6000    (30500 cumulative)
#   VSSD-gamma       on_distill_vssd_gamma/ckpt_9000  (9000)
#   bidirectional    on_distill_bidir/ckpt_9000       (9000)
# So compare each arm against its own previous number, not against each other.
#
# Plateau schedule with a real held-out split, at 5e-5. Not 3e-4: these are continuations of
# converged mixers, and re-heating one is what produced the 0.1344 -> 0.1561 -> 0.2105 slide.
#
# Budget from ~21:30 (when the rope ablation ends) at measured 0.85-0.93 s/step:
#   3 x (10000 Phase-B + 1000 Phase-C + eval) ~= 8.4 h -> finishes ~06:00, before 07:00.
# Ordered as requested, and an arm that fails is skipped rather than taking the rest with it.
set -uo pipefail
cd "$(dirname "$0")/.."

# Do not contend with the rope ablation for the GPU; serial is faster here anyway
# (concurrent runs measured 0.89x aggregate throughput).
while pgrep -f "run_bidir_no[_]rope_all" > /dev/null; do sleep 60; done
sleep 30

STEPS=${STEPS:-10000}

run_arm () {
  local tag="$1" variant="$2" init="$3" label="$4"
  local dist="result/runs/fl_distill_$tag" ft="result/runs/fl_ft_$tag"
  mkdir -p "$dist" "$ft"

  if [ ! -f "$init" ]; then echo "[$tag] missing init $init -- skipped"; return 1; fi

  echo "=== [$tag] Phase-B $STEPS from $init, first-layer RoPE ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 1 --sub 1 --variant "$variant" --rope-first-layer \
      --init-ckpt "$init" \
      --scheduler plateau --val-scenes eth3d:relief_2,eth3d:electro \
      --test-every 1000 --test-n-views 4 --test-n-batches 4 \
      --plateau-factor 0.5 --plateau-patience 2 --plateau-min-lr 1e-6 --plateau-metric L_D \
      --warmup-steps 100 --steps "$STEPS" --ckpt-every 2000 --lr-attn 5.0e-5 \
      --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "[$tag] Phase-B FAILED"; return 1; }

  local ck; ck=$(ls -t "$dist"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$ck" ] || { echo "[$tag] no checkpoint"; return 1; }

  echo "=== [$tag] Phase-C from $ck ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 3 --sub 3 --variant "$variant" --rope-first-layer --init-ckpt "$ck" \
      --scheduler plateau --val-scenes eth3d:relief_2,eth3d:electro \
      --test-every 250 --test-n-views 4 --test-n-batches 4 \
      --plateau-factor 0.5 --plateau-patience 2 --plateau-min-lr 1e-7 --plateau-metric L_D \
      --warmup-steps 50 --steps 1000 \
      --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
      --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "[$tag] Phase-C FAILED"; return 1; }

  echo "=== [$tag] eval ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
      --ckpt "$ft/ckpt_1000.pt" --label "$label" \
      --out "result/depth_eval_fl_$tag.json" 2>&1 | grep -E "abs_rel=" || true
}

run_arm vssd_bg vssd_bg result/runs/ext_distill_vssd_bg/ckpt_6000.pt \
        "VSSD-beta,gamma (first-layer RoPE, +10k)"
run_arm vssd_gamma vssd result/runs/on_distill_vssd_gamma/ckpt_9000.pt \
        "VSSD-gamma (first-layer RoPE, +10k)"
run_arm bidir mamba3 result/runs/on_distill_bidir/ckpt_9000.pt \
        "bidirectional (first-layer RoPE, +10k)"

echo "=== all arms done ($(date -Is)) ==="
grep -h abs_rel result/depth_eval_fl_*.json 2>/dev/null || true
echo "--- references: rope-all-layers 9k: VSSD-b,g 0.1153 | VSSD-g 0.1964 | bidir 0.2036 ---"
echo "--- DA3-SMALL finetuned 0.0362, zero-shot 0.0334, CM22 legacy 0.053 ---"
