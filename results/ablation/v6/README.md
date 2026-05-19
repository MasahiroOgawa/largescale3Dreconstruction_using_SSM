# v6 — BFA (loss-only fix): Smooth-L1 + dir-cosine + scaled mag + 2D reproj

## What this run tests

Building on v5 (which collapsed to a "predict near-zero motion" trivial baseline
under scale-normalised L1), v6 adds four new loss terms to attack that baseline
without changing the architecture:

- **Smooth-L1 / Huber (δ=1)** on the scaled relative-position residual `(Δp̂ − Δp*) / s`.
- **Direction cosine** `1 − cos(Δp̂, Δp*)` — penalises wrong-direction motion regardless of magnitude.
- **Scaled magnitude** `((‖Δp̂‖ − ‖Δp*‖) / s)²` — enforces the right motion-to-scene-scale ratio (the only magnitude info recoverable from monocular RGB).
- **2-D pixel reprojection** Smooth-L1 on `π(p_q + Δp̂, K) − π(p*, K)` divided by `image_size`.

`λ_pos, λ_mag, λ_dir, λ_reproj, λ_vis, λ_spawn, λ_smooth = (1, 0.5, 0.5, 1, 0.5, 0.5, 0.1)`.

Otherwise identical to v5: same `Mamba3Tracker` (4.28 M params, level_sizes=(32, 64)), same 30 k-step fp32 schedule, same data (pstudio + drivetrack + adt). Implementation commit: `606066a`.

## Outcome — same predict-zero baseline as v5

| | v5 (raw L1) | **v6 (BFA loss)** | Note |
|---|---|---|---|
| val pos | 0.017 | **0.0006** | Smooth-L1 makes the tiny-residual region cheaper, model goes even more zero |
| val mag | — | 0.0006 | Both pred and GT magnitudes are tiny |
| val dir | — | **1.03** | cos ≈ −0.03 — predicted direction is essentially perpendicular / random |
| val reproj | — | 0.0002 | Pixel-space agrees (because abs position ≈ p_query GT) |
| val vis | 0.030 | 0.099 | Slightly worse |
| val spawn | 0.0064 | 0.010 | Comparable |
| `‖Δp̂‖_mean` | ~4 mm | ~few mm | Predict-near-zero |
| `‖Δp*‖_mean` | ~41 cm | ~41 cm | GT motion unchanged |

The direction-cosine term (which v5 lacked) makes the failure visible — but doesn't fix it. The model settled into "tiny vector in random direction", where:
- pos / mag / reproj are all tiny because both sides are tiny;
- direction-cosine alone signals "wrong" but its gradient at near-zero predicted magnitude is numerically unstable; the gradient direction flips between batches.

Training was stopped early at **step 4000 / 30000** after val direction-cosine bounced in 0.91 – 1.16 across eight val checkpoints with no learning signal. Continuing 30 k would not change the conclusion.

## TAPVid-3D evaluation

`ckpt_4000.pt` on the 3 subsets (407 clips total; ADT eval OOM'd on full-clip inference, fixed in v7 with chunked inference):

| subset | 3D-AJ | APD3D | OA |
|---|---|---|---|
| pstudio | 0.0077 | 0.0077 | 0.7581 |
| drivetrack | 0.0013 | 0.0013 | 0.7890 |
| adt | n/a (OOM) | n/a (OOM) | n/a (OOM) |
| **mean (pstudio+drivetrack)** | **0.0045** | **0.0045** | **0.7735** |

Compared to published baselines on the same minival benchmark
(`configs/tapvid3d_baselines.yaml`):

| method | aria | drivetrack | pstudio | mean |
|---|---|---|---|---|
| BootsTAPIR + ZoeDepth | 0.123 | 0.075 | 0.213 | 0.137 |
| SpatialTracker | 0.137 | 0.094 | 0.225 | 0.152 |
| CoTracker3D-online | 0.146 | 0.107 | 0.213 | 0.155 |
| CoTracker3D-offline | 0.163 | 0.135 | 0.222 | 0.173 |
| DELTA | 0.176 | 0.130 | 0.244 | 0.183 |
| **v6 (this run)** | n/a | 0.001 | 0.008 | 0.005 |

Position-tracking metrics are effectively zero (consistent with the predict-zero
failure mode); occlusion accuracy is real because the visibility head trained
correctly off the standalone BCE.

## What this tells us about the next variant

- Loss-only fixes are **insufficient** when the underlying architecture has no
  signal for "where did this point move to in frame *t+1*". The cross-attention
  in this implementation uses a rank-1 decay mask (see `mamba3_attn/mamba3/cross_attention.py`)
  which cannot compute per-query-per-key adaptive weighting.
- **v7 = C + D** (iterative refinement + explicit correlation-volume features)
  directly addresses this: each query is told "places in the new frame that
  look like me" as an attention bias / extra input, instead of having to
  rediscover that signal through gradient descent.

## Files in this directory

- `eval_summary.md` — raw output of `scripts/eval_mamba3_tracker.py`.
- `comparison.md` — head-to-head with published baselines.
- `training_curve.png` — 6-panel loss curve over the 4 k step run.
- `metric_results/{pstudio,drivetrack,adt}.json` — per-clip metrics.
- Implementation: commit `606066a` (`mamba3_tracker/train/loss.py`,
  `mamba3_tracker/data/dataset.py`, `scripts/train_mamba3_tracker.py`).
- TeX doc: `doc/attention/mamba3_attention.tex §8.6` updated in the same commit.
