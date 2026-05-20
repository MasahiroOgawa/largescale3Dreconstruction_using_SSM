# v8 — F: velocity-based loss to match the TAPVid-3D eval metric

## What this run tests

v7's qualitative videos showed the predicted tracks remained visually
**static** even though the 3D-AJ headline jumped 9.7× over v6 — points
weren't tracking the GT motion. The cause was a loss / eval mismatch:

- v6/v7 loss penalised `(Δp̂ − Δp*) / s` with per-clip median scene
  scale `s ≈ 1 m`. For pstudio motion ≈ 5 cm, the scaled residual
  ≈ 0.05; Smooth-L1(0.05) ≈ 0.00125. **Predicting Δp̂ = 0 already sits
  at the loss floor — the optimiser had almost no gradient.**
- The TAPVid-3D evaluator measures `‖p̂_t − p*_t‖` in absolute metres
  against multiple thresholds. With `p̂_t = p*_anchor + Δp̂_t`, the
  static-zero predictor `Δp̂ = 0` makes `p̂_t = p*_anchor`, which is
  inside the larger thresholds for the bulk of frames in short clips.
  **So the eval also rewards "predict static at the anchor"** when GT
  motion is small relative to scene scale.

v8 redesigns the loss to break both incentives:

  * **Velocity residual** `(v̂(t) − v*(t)) / s_3D(t, n)` and its 2D
    pixel-space analogue `(û − u*) / s_2D`. Velocity makes a static
    Δp̂ produce constant non-zero error wherever GT moves — collapse
    is no longer free.
  * **Position residual** `(p̂(t) − p*(t)) / s_3D(t, n)` and 2D
    analogue, with smaller weight (0.1). Prevents per-track DC-drift
    that pure-velocity loss would tolerate.
  * **Huber-clipped per-(t, n) scale**
    `s_3D(t, n) = sqrt(δ_3D² + ‖v*(t, n)‖²)`, δ_3D = 5 cm;
    `s_2D(t, n) = sqrt(δ_2D² + ‖u*(t, n)‖²)`, δ_2D = 1 px.
    Floors at δ to keep the loss informative on slow tracks; grows
    with GT velocity for fast tracks.
  * **Smoothness on the residual** (not on the prediction) — so the
    static predictor cannot collect a free 0 smoothness.
  * **Visibility BCE** kept (surrogate for Occ-Acc). Spawn loss
    **removed** (not in TAPVid-3D eval).
  * **Config-driven weights normalised to Σ = 1** read from a single
    unified YAML at `configs/v8.yaml` (sections `model:`, `data:`,
    `train:`, `loss:`).

Same v7 architecture (`CausalCrossPropagator` with iterative
refinement + correlation cross-attention), same data, same schedule.
The only deltas vs v7 are the loss + bf16 precision (forced by GPU
sharing with another active training process; documented).

## Outcome — pstudio improved, drivetrack regressed

| subset | v7 3D-AJ | **v8 3D-AJ** | change |
|---|---|---|---|
| pstudio          | 0.0769 | **0.0961** | **+25 %** |
| drivetrack       | 0.0104 | **0.0020** | **−81 %** |
| adt (max64)      | 0.0908 | 0.0768     | −15 % |
| (p+d) mean       | 0.0437 | **0.0490** | +12 % |
| 3-subset mean    | 0.0594 | 0.0583     | −2 % |

Mixed. The (pstudio + drivetrack) mean improved by 12 %, but the
underlying movement is large and asymmetric:

- **Pstudio (+25 %).** Slow indoor motion is the regime where v6/v7's
  scale-normalised loss had near-zero gradient. v8's δ-floored scale
  (5 cm for 3D, 1 px for 2D) keeps the gradient alive on these
  small-motion frames. This is the case the redesign was *built* for
  and it pays off.

- **Drivetrack (−81 %).** Fast camera-pan motion (tens of pixels per
  frame). With `s_2D ≈ ‖u*‖` for large `‖u*‖`, `1/s_2D` is small —
  **the 2D loss effectively turns off where the motion is largest.**
  The 3D term is also weakened by the same effect on `s_3D ≈ ‖v*‖`.
  Predicted by the v8 design review beforehand; not mitigated in this
  run to keep the v8 spec literal.

- **ADT (−15 %).** ADT has a mix of slow-hand and fast-pan motion
  segments. Net loss is consistent with the drivetrack pattern at
  smaller magnitude.

### Loss curves show the redesign is working as designed

Throughout 30 k bf16 steps:

- `vel_3D ≈ 0.07–0.1` and `vel_2D ≈ 0.4–0.5` on val — the model is
  *learning velocity*, not floor-pegged.
- `pos_3D ≈ 0.4–2.0` and `pos_2D ≈ 5–30 px` on val — meaningful
  per-frame position residuals (vs. v6/v7's val pos ≈ 0.0003, which
  was the predict-zero signature).
- Per-step training loss spikes (`p3D = 100–600`) on clips where the
  model momentarily slips back toward static — exactly the failure
  mode the small-δ scale was designed to penalise. The spikes don't
  blow up gradients because of grad-clip 1.0 + NaN-guard; no
  NaN-guard hits across all 30 k steps.

### Visual qualitative result

`outputs/eval_tracker/v8/viz/*.mp4` — 2 clips per subset, 64 frames
each. **Primary acceptance criterion is whether the points actually
move now.** The training/val numbers show motion is being learned;
the videos are the user-facing confirmation.

## What this tells us about the next variant

- The δ-floored scale was the right idea for slow motion but **the
  unbounded upper end blunts the loss on fast motion**. v8's drivetrack
  regression is the direct consequence. The plan agent flagged this
  upfront and the strict-spec-first interpretation didn't mitigate it.
- **v9 should add a ceiling**: `s_2D = clamp(sqrt(δ² + ‖u*‖²), 1 px,
  δ_hi)` with δ_hi ≈ image_size / 32 ≈ 14 px (analogous bound for 3D
  at δ_hi ≈ 0.5 m). That should recover drivetrack without harming
  pstudio.
- The originally-planned **v9 = E (frozen DINOv2 encoder)** is still
  valid but probably should wait until the v8 fast-motion regression
  is fixed first — otherwise we won't be able to isolate whether
  changes in v10 come from the encoder or from the scale ceiling.
- Open since v6: full-clip ADT eval still OOMs. A chunked-encoder
  inference path is the right fix; not in scope this run.

## Training run details

- 30 000 steps, **bf16** (vs. v7's fp32), 4 h 19 min on the 11.6 GB
  GPU while sharing the device with a 2.3 GB gsplat training process.
- No NaN-guard hits across all 30 k steps.
- Two-launch story: the first attempt at `--amp fp32` CUDA-OOM'd at
  step 1 because of the gsplat contention; relaunched with `--amp
  bf16` (the YAML default stays fp32 — bf16 was a CLI override).
- Training command:
  ```bash
  systemd-run --user --scope --quiet \
      -p MemoryMax=18G -p MemorySwapMax=8G -p OOMPolicy=continue \
      bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          uv run python scripts/train_mamba3_tracker.py \
          --config configs/v8.yaml \
          --out-dir outputs/runs/mamba3_tracker_v8 \
          --amp bf16'
  ```

## Files in this directory

- `README.md` — this file.
- `eval_summary.md` — per-subset table.
- `comparison.md` — v6 → v7 → v8 ablation + against published baselines.
- `training_curve.png` — 8-panel loss curves over the 30 k bf16 run.
- `metric_results/{pstudio, drivetrack, adt}.json` — per-clip metrics.
- Implementation: `src/mamba3_tracker/train/loss.py`,
  `src/mamba3_tracker/train/config.py`, `configs/v8.yaml`,
  `tests/unit/test_tracking_loss_v8.py`,
  `scripts/train_mamba3_tracker.py` rewrite.
