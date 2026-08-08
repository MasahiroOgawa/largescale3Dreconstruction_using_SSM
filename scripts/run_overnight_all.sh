#!/usr/bin/env bash
# Everything needed for the paper by 10:00: the three ETH3D depth arms, then the four
# CIFAR-10 fixed-turn cells. One chain so nothing contends for the GPU -- concurrent runs
# measured 0.89x aggregate throughput here against a 2.0x break-even, so serial is faster
# in wall-clock as well as safer on 11.6 GB.
#
# Budget from 18:10 to 10:00 is 15.8 h. Measured rates: ~1.16 s/step for VSSD-beta,gamma,
# ~1.0 for the one-pool operators, ~255 s/epoch for CIFAR at T=1025.
#
#   CIFAR   4 runs x 30 epochs                       8.5 h   (30 epochs is fixed: fewer
#                                                             would not be comparable with
#                                                             the rest of Table 1)
#   ETH3D   Phase C 1000 + eval, three arms          1.0 h
#   ETH3D   Phase B 9000 x 3                          7.9 h   (9000 is the stated floor)
#
# 17.4 h total, finishing about 11:35 -- past 10:00, accepted rather than drop below 9000.
# 9000 is still under CM12/CM24's 20000, the budget that produced 0.0513, so a result short
# of that number is not by itself evidence against the operator.
#
# ETH3D runs first, ours first within it: if anything slips it costs the last CIFAR cell,
# which is the most replaceable thing here.
set -uo pipefail
cd "$(dirname "$0")/.."
VM3=/home/mas/proj/study/visionMamba3

STEPS=${STEPS:-9000}

for entry in "vssd_bg:VSSD-beta,gamma" "vssd:VSSD-gamma" "mamba3:bidirectional"; do
  variant="${entry%%:*}"; label="${entry#*:}"
  case "$variant" in
    vssd_bg) tag=vssd_bg ;;
    vssd)    tag=vssd_gamma ;;
    mamba3)  tag=bidir ;;
  esac
  dist="result/runs/on_distill_$tag"; ft="result/runs/on_ft_$tag"
  mkdir -p "$dist" "$ft"

  echo "=== [$tag] Phase-B $STEPS steps, cosine, DA3-SMALL teacher ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 1 --sub 1 --variant "$variant" --rope-all-layers \
      --scheduler cosine --warmup-steps 200 --steps "$STEPS" --ckpt-every 2000 \
      --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "[$tag] Phase-B FAILED"; continue; }

  # Newest checkpoint, not a fixed name: an arm cut short still gets a Phase C and a number.
  init=$(ls -t "$dist"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$init" ] || { echo "[$tag] no checkpoint"; continue; }

  echo "=== [$tag] Phase-C from $init ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python -m mamba3_attn.train.train_super \
      --super 3 --sub 3 --variant "$variant" --rope-all-layers --init-ckpt "$init" \
      --scheduler cosine --warmup-steps 100 --steps 1000 \
      --lr-attn 1.0e-5 --lr-head 1.0e-5 --lr-other 3.0e-5 \
      --out-dir "$ft" 2>&1 | tee "$ft/train.log" || { echo "[$tag] Phase-C FAILED"; continue; }

  echo "=== [$tag] eval ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_depth_metrics.py \
      --ckpt "$ft/ckpt_1000.pt" --label "$label (cosine, $STEPS)" \
      --out "result/depth_eval_on_$tag.json" 2>&1 | grep -E "abs_rel=" || true
done

echo "=== ETH3D arms done ($(date -Is)); DA3-SMALL reference abs_rel 0.0362 ==="
grep -h abs_rel result/depth_eval_on_*.json 2>/dev/null || true

echo "=== CIFAR-10 fixed-turn grid ($(date -Is)) ==="
cd "$VM3"
bash src/eval/run_rope_turns_grid.sh || echo "CIFAR grid did not complete"

echo "=== everything done ($(date -Is)) ==="
