# TAPVid-3D comparison — v7

## 3D-AJ per subset (higher is better)

| method | aria/adt | drivetrack | pstudio | mean |
|---|---|---|---|---|
| BootsTAPIR + ZoeDepth | 0.123 | 0.075 | 0.213 | 0.137 |
| SpatialTracker        | 0.137 | 0.094 | 0.225 | 0.152 |
| CoTracker3D-online    | 0.146 | 0.107 | 0.213 | 0.155 |
| CoTracker3D-offline   | 0.163 | 0.135 | 0.222 | 0.173 |
| DELTA                 | 0.176 | 0.130 | 0.244 | 0.183 |
| **Mamba-3 v6**        | n/a (OOM)  | 0.0013     | 0.0077     | 0.0045 |
| **Mamba-3 v7 (this run)** | **0.091 ⚠** | **0.0104** | **0.0769** | **0.059** |

⚠ ADT eval was run with `--max-frames 64` because the encoder OOMs on
full ADT clips (~300 frames at 512×512). Baselines evaluate on full
clips — this v7 ADT number is *partial-clip* and not directly
comparable. The pstudio / drivetrack numbers ARE full-clip.

## v5 → v6 → v7 ablation (higher 3D-AJ is better)

| variant | architecture / loss change                                               | drivetrack | pstudio | mean |
|---|---|---|---|---|
| v5 | rank-1 cross-mask SSD only, raw L1 (scale-normed)                        | ~0.001 (predict-zero) | ~0.008 (predict-zero) | ~0.005 |
| v6 | + Smooth-L1, dir-cosine, scaled mag, 2D reprojection loss                | 0.0013     | 0.0077     | 0.0045 |
| **v7** | **+ correlation-volume cross-attention, + iterative refinement (3 iters)** | **0.0104** | **0.0769** | **0.0437** |

**Headline:** v7 is ~10× v6 on both pstudio (10.0×) and drivetrack (8.0×).
The C+D architectural change is the first thing that actually moved the
TAPVid-3D needle. Loss-only fixes (v6) did not.

## Per-subset detail of this run

| subset | 3D-AJ | APD3D | OA |
|---|---|---|---|
| pstudio    | 0.0769 | 0.0769 | 0.7581 |
| drivetrack | 0.0104 | 0.0104 | 0.7932 |
| adt (first 64 frames) | 0.0908 | 0.0908 | 0.5536 |

Baseline numbers from `configs/tapvid3d_baselines.yaml`
(TAPVid-3D paper + each method's released numbers).
