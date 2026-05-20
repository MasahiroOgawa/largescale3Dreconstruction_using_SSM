# TAPVid-3D comparison — v8

## v6 → v7 → v8 ablation (higher 3D-AJ is better)

| variant | loss / arch change                                                      | drivetrack | pstudio | mean (p+d) |
|---|---|---|---|---|
| v6      | Smooth-L1 + dir-cosine + scaled mag + 2D reproj on scale-normed Δp     | 0.0013     | 0.0077     | 0.0045   |
| v7      | + iterative refinement + correlation-volume cross-attention             | 0.0104     | 0.0769     | 0.0437   |
| **v8**  | **+ velocity & position w/ Huber-scaled, normalised weights, unified YAML** | **0.0020** | **0.0961** | **0.0490** |

v8 changes vs v7:
- **Loss redesigned around velocity + position** (`(v − v_GT)/s` and `(p − p_GT)/s`).
- **Single Huber-clipped scale** `s = sqrt(δ² + ‖v_GT‖²)` per (t, n) per domain.
  δ_3D = 5 cm, δ_2D = 1 px.
- **Smoothness on the residual** (`((v̂ − v*) − (v̂_{-1} − v*_{-1}))/s`), not
  the prediction — so the static-zero predictor cannot get a free 0.
- **Spawn loss removed** (not in TAPVid-3D eval).
- **Config-driven normalised weights** (`Σ λ_i = 1`).
- **bf16** (instead of v7's fp32) because the GPU was being shared with
  another training process; recipe deviation noted.

## 3D-AJ vs published baselines

| method | aria/adt | drivetrack | pstudio | mean |
|---|---|---|---|---|
| BootsTAPIR + ZoeDepth | 0.123 | 0.075 | 0.213 | 0.137 |
| SpatialTracker        | 0.137 | 0.094 | 0.225 | 0.152 |
| CoTracker3D-online    | 0.146 | 0.107 | 0.213 | 0.155 |
| CoTracker3D-offline   | 0.163 | 0.135 | 0.222 | 0.173 |
| DELTA                 | 0.176 | 0.130 | 0.244 | 0.183 |
| Mamba-3 v6            | n/a (OOM) | 0.0013 | 0.0077 | 0.0045 |
| Mamba-3 v7            | 0.091 ⚠   | 0.0104 | 0.0769 | 0.059 |
| **Mamba-3 v8 (this run)** | **0.077 ⚠** | **0.0020** | **0.0961** | **0.058** |

⚠ ADT eval with `--max-frames 64` because encoder OOMs on full ADT
clips. Pstudio / drivetrack are full-clip.

## Per-subset detail of this run

| subset | 3D-AJ | APD3D | OA |
|---|---|---|---|
| pstudio    | 0.0961 | 0.0961 | 0.7581 |
| drivetrack | 0.0020 | 0.0020 | 0.7890 |
| adt (first 64 frames) | 0.0768 | 0.0768 | 0.5536 |

Baseline numbers from `configs/tapvid3d_baselines.yaml`.
