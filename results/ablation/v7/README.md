# v7 — CD: iterative refinement + correlation-volume cross-attention

## What this run tests

After v6 confirmed that loss-only fixes could not lift the model out of
the "predict near-zero motion" trivial baseline (the rank-1 decay mask
of `Mamba3CrossAttention` does not give per-(query, key) adaptive
weighting, so SSD cross-attention had no native way to *find* matching
features), v7 changes the architecture:

- **(D) Correlation-volume cross-attention.** At every pyramid level of
  every frame we compute `corr[n, j] = cos(Q[n], kv_token[j]) / τ`, soft-
  max over the kv axis, and weighted-sum kv tokens. The resulting
  `(B, N, D)` increment is summed with the existing variant-B SSD output
  — a residual two-branch cross-attention (`SSD + corr`) at each level.
  Each query gets an explicit *"places in this frame that look like
  me"* signal, instead of having to discover that signal through
  gradient descent.
- **(C) Iterative refinement.** Each frame runs the coarse→fine pyramid
  pass `N_ITER = 3` times. The refined bank from iteration *i* is the
  input to iteration *i+1*, so the model can re-attend to the same kv
  features with an updated query — RAFT-style.

Loss is unchanged from v6 (Smooth-L1 pos + scaled mag + dir-cosine + 2D
reprojection + vis/spawn BCE + smoothness). 30 k steps, fp32, same
data (pstudio + drivetrack + adt). Implementation lands in this commit.

## Outcome — ~10× v6 on TAPVid-3D 3D-AJ

| variant | drivetrack 3D-AJ | pstudio 3D-AJ | (drivetrack+pstudio) mean |
|---|---|---|---|
| v5 | ~0.001 | ~0.008 | ~0.005 |
| v6 | 0.0013 | 0.0077 | 0.0045 |
| **v7** | **0.0104 (8.0×)** | **0.0769 (10.0×)** | **0.0437 (9.7×)** |

C + D is the **first architectural change in the v5→v6→v7 sequence that
actually moves the TAPVid-3D needle.** Loss-only fixes (v6) did not.

### The training/val signal didn't predict this

Through all 30 k training steps, validation direction-cosine bounced
between 0.79 and 1.14 (val rows every 500 steps) — *the same predict-
near-zero pattern v6 exhibited*. The 6-panel `training_curve.png` looks
qualitatively identical to v6's. If we had stopped early on the val
signal alone (as v6 did at step 4 000), we would have missed this.

The reason: training metrics are computed on **8-frame windows** with
random anchor offsets. GT motion in 8 frames is small, so "predict zero
motion" is numerically a near-optimal answer for pos / mag /
reproj — but produces a random, near-perpendicular direction (cos ~0)
because the gradient on `1 − cos(Δp̂, Δp*)` is unstable when
`‖Δp̂‖ ≈ 0`. The TAPVid eval, however, integrates over full clips and
threshold-checks position closeness — that's where the correlation-volume
attention's *gradient information* materialises, even though the training
loss never strongly registered it.

## TAPVid-3D evaluation

`ckpt_30000.pt` on all 3 subsets:

| subset | 3D-AJ | APD3D | OA | mode |
|---|---|---|---|---|
| pstudio    | 0.0769 | 0.0769 | 0.7581 | full clips |
| drivetrack | 0.0104 | 0.0104 | 0.7932 | full clips |
| adt        | 0.0908 | 0.0908 | 0.5536 | first 64 frames (see caveat) |
| **mean (pstudio + drivetrack)** | **0.0437** | **0.0437** | **0.7756** |  |
| **mean (3 subsets)**            | **0.0594** | **0.0594** | **0.7016** |  |

**ADT caveat.** ADT clips are ~300 frames at 512×512; full-clip
inference OOMs the 11.6 GB GPU during encoder activation, even with
`@torch.no_grad()` + bf16 (this was also the v6 failure mode). We capped
ADT at `--max-frames 64` to produce a number. Pstudio / drivetrack
ARE full-clip and ARE directly comparable to published baselines.
A chunked-encoder eval is the proper fix; deferred to v8.

### Against published baselines (`configs/tapvid3d_baselines.yaml`)

| method | aria/adt | drivetrack | pstudio | mean |
|---|---|---|---|---|
| BootsTAPIR + ZoeDepth | 0.123 | 0.075 | 0.213 | 0.137 |
| SpatialTracker        | 0.137 | 0.094 | 0.225 | 0.152 |
| CoTracker3D-online    | 0.146 | 0.107 | 0.213 | 0.155 |
| CoTracker3D-offline   | 0.163 | 0.135 | 0.222 | 0.173 |
| DELTA                 | 0.176 | 0.130 | 0.244 | 0.183 |
| **v7 (this run)**     | 0.091 ⚠ | 0.0104 | 0.0769 | 0.059 |

Still ~3× below the weakest baseline on drivetrack and ~2.8× below on
pstudio, but for the first time *on the same order of magnitude* — v6
was 1–2 orders of magnitude below.

## What this tells us about the next variant

- Architectural matching signal (correlation-volume) was the load-
  bearing missing piece, *not* the loss recipe. Future variants should
  keep the v7 architecture and iterate from here.
- Val direction-cosine is a misleading early-stop signal for short
  random-window training. Drive future early-stopping decisions off
  *TAPVid-3D mid-training eval*, not val loss alone.
- v8 = **E** (frozen-DINOv2-pretrained encoder) is the next planned
  step. Likely complementary to v7's correlation cross-attention because
  it gives the kv side a stronger initial feature space, on which
  cosine-similarity already encodes "looks like me".
- An orthogonal issue surfaced by this run: the encoder OOMs on long
  ADT clips during eval. Worth fixing before v8 eval so we can drop the
  partial-clip caveat.

## Training run details

- 30 000 steps, fp32, ~5 h 11 min on a single 12 GB GPU.
- Two earlier launches died at step ~200 to systemd-oomd; root cause
  was a dataloader RAM leak (`load_clip` decoded whole drivetrack clips
  into RAM, ~566 MB each, kept alive across the run by
  `persistent_workers=True`). Fixed in commit `3c0a268` —
  `load_clip(path, frames=(s, e))` + `peek_clip_F(path)` +
  `persistent_workers=False`. RSS smoke after the fix stays flat at
  ~619 MB across 40 calls.
- Training command (after fix):
  ```bash
  systemd-run --user --scope --quiet \
      -p MemoryMax=18G -p MemorySwapMax=8G -p OOMPolicy=continue \
      bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          uv run python scripts/train_mamba3_tracker.py \
          --out-dir outputs/runs/mamba3_tracker_v7 \
          --subsets pstudio drivetrack adt \
          --steps 30000 \
          --warmup 1000 --decay 3000 \
          --ckpt-every 1000 --val-every 500 --log-every 50 \
          --lr 2e-4 --weight-decay 0.05 --grad-clip 1.0 \
          --window 8 --batch 2 --image-size 448 \
          --num-tracks 256 --level-sizes 32 64 \
          --num-heads 6 --state-dim 64 --dim 384 \
          --amp fp32 --num-workers 1 --seed 0'
  ```

## Files in this directory

- `README.md` — this file.
- `eval_summary.md` — per-subset table.
- `comparison.md` — against published baselines + ablation row.
- `training_curve.png` — 8-panel loss curve over the 30 k step run.
- `metric_results/{pstudio,drivetrack,adt}.json` — per-clip metrics.
- Implementation: this commit (`src/mamba3_tracker/model/propagator.py`).
- ADT was evaluated in a separate invocation with `--max-frames 64`;
  the resulting JSON is `metric_results/adt.json`.
