#!/usr/bin/env bash
# All three operators, DA3-SMALL teacher, cosine schedule, fresh Phase B -> Phase C -> eval.
#
# Fresh rather than continued. Every warm-started continuation ran Phase B at 3e-4 because
# scope="attn" ignored --lr-attn, re-heating an already-converged mixer; depth degraded
# monotonically (0.1344 -> 0.1561 -> 0.2105), so those checkpoints are damaged rather than
# unfinished and continuing from them would inherit the damage. CM12/CM24's recipe -- fresh
# mixer, DA3-SMALL teacher, 3e-4 -- is the only one here that has produced a good number
# (0.0513), so this reproduces it with the cosine shape on top. 3e-4 is correct for a fresh
# mixer; it was only wrong as a continuation rate.
#
# Cosine decays from the first post-warmup step instead of holding the peak like WSD, so the
# final checkpoint comes from a settled model rather than mid-oscillation -- the failure the
# feature-loss trace showed (0.264 -> 0.352 -> 0.280 -> 0.335 -> 0.359 inside one stage).
#
# 15000 steps is sized to the window: measured ~1.16 s/step for VSSD-beta,gamma and ~1.0 for
# the one-pool operators, so 4.83 + 4.17 + 4.17 = 13.2 h, plus ~19 min of Phase C and eval
# per arm, finishing about 08:10 against a 09:00 deadline. Inside the 9k-20k range asked for.
# Ours runs first, so any slip costs the arm that is already citable.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-15000}

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
      --scheduler cosine --warmup-steps 200 --steps "$STEPS" --ckpt-every 3000 \
      --out-dir "$dist" 2>&1 | tee "$dist/train.log" || { echo "[$tag] Phase-B FAILED"; continue; }

  # Newest checkpoint rather than a fixed name, so an arm cut short by the deadline still
  # gets a Phase C and an evaluated number instead of failing on a missing file.
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

echo "=== all three arms done ($(date -Is)) ==="
echo "--- DA3-SMALL reference: abs_rel 0.0362, delta<1.25 0.9996 ---"
grep -h abs_rel result/depth_eval_on_*.json 2>/dev/null || true
