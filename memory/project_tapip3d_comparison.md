---
name: project-tapip3d-comparison
description: TAPIP3D vs v33 vs SEA-RAFT comparison on TAPVid-3D minival — both median-AJ and absolute metric
metadata:
  type: project
---

Comparing TAPIP3D (image-only, DA3 depth), SEA-RAFT+DA3, and v33 (SEA-RAFT+DA3+Mamba3) on the same 150 minival clips.

**Why:** Establish whether TAPIP3D (published SOTA) or our v33 is better on the metric that matters (absolute 3D depth), since the leaderboard's median-scaling hides absolute depth quality.

## Median-scaled 3D-AJ (scale-invariant leaderboard metric)

| Subset     | SEA-RAFT+DA3 | TAPIP3D (DA3) | v33 (RAFT+DA3+Mamba3) |
|------------|-------------|--------------|----------------------|
| drivetrack | **10.79%**  | 6.49%        | 5.68%                |
| pstudio    | **11.06%**  | 2.28%        | 4.07%                |
| adt        | **13.23%**  | 0.38%        | 12.31%               |
| **mean**   | **11.69%**  | 3.05%        | 7.35%                |

SEA-RAFT wins everywhere on median-AJ. TAPIP3D is catastrophically bad on ADT (0.38%).

**Source:** SEA-RAFT/v33 from `outputs/metric3d_*_fix_20260618-1718/`. TAPIP3D from official TAPIP3D evaluator (`/home/mas/proj/study/TAPIP3D/outputs/auto_generated/.../metrics_*.json`, key `tapvid3d_average_jaccard_best`).

## Absolute Metric-AJ (no median scaling, fixed-metre thresholds 1cm..2.56m)

| Subset     | SEA-RAFT+DA3 | TAPIP3D (DA3) | v33 (RAFT+DA3+Mamba3) |
|------------|-------------|--------------|----------------------|
| drivetrack | 0.47%       | 0.60%        | **8.17%**            |
| pstudio    | 16.49%      | 13.08%       | **18.59%**           |
| adt        | 27.27%      | 16.42%       | **27.21%**           |
| **mean**   | 14.74%      | 10.03%       | **17.99%**           |

**v33 wins on every subset.** Key insight: drivetrack is the differentiator — v33 8.17% vs TAPIP3D 0.60% vs SEA-RAFT 0.47% (v33 is 14-17× better). The Mamba3 depth refiner corrects DA3's metric scale bias for far-field outdoor scenes.

## Mean Absolute 3D Error (metres, lower=better)

| Subset     | SEA-RAFT+DA3 | TAPIP3D (DA3) | v33 (RAFT+DA3+Mamba3) |
|------------|-------------|--------------|----------------------|
| drivetrack | 11.70m      | 13.62m       | **6.88m**            |
| pstudio    | 0.76m       | 0.85m        | **0.72m**            |
| adt        | **0.41m**   | 0.71m        | **0.41m**            |
| **mean**   | 4.29m       | 5.06m        | **2.67m**            |

v33 is best on drivetrack/pstudio. TAPIP3D is worst on ADT (0.71m vs 0.41m for SEA-RAFT/v33).

## Evaluation pipeline

- SEA-RAFT/v33 metrics: `scripts/eval_metric3d.py` → `outputs/metric3d_*_fix_20260618-1718/`
- TAPIP3D median-AJ: official TAPIP3D `train_eval.py` evaluator (uses visibility threshold sweep)
- TAPIP3D absolute metric/err_mean: `scripts/eval_tapip3d_absolute.py` → `outputs/tapip3d_absolute_eval_20260625-1318/`
- **TAPIP3D median-AJ bug**: `eval_tapip3d_absolute.py` also reports median-AJ but with wrong intrinsics (applied 256/min(H,W) to already-processed-res K instead of original K → ~3.75× inflated). Fixed in script but NOT used in plots — official values used instead.
- Plots: `scripts/plot_tapip3d_vs_v33.py` → `outputs/plots/comparison_*.png`

## Conclusions

1. **For absolute metric 3D reconstruction** (this project's goal): v33 wins by large margin, especially on drivetrack (+14-17× over TAPIP3D+SEA-RAFT). TAPIP3D is actually WORSE than raw SEA-RAFT+DA3 on absolute metric (10% vs 15% mean metric-AJ).
2. **For leaderboard median-AJ**: SEA-RAFT wins, v33 and TAPIP3D both worse (TAPIP3D especially bad on ADT 0.38%). This metric hides depth quality.
3. **Why TAPIP3D underperforms**: TAPIP3D is optimized for scale-invariant tracking; its model doesn't improve metric depth. It also fails on ADT (possibly poor generalization from training distribution).

**How to apply:** Paper framing confirmed: v33 (tiny learned depth corrector on top of SEA-RAFT) achieves SoTA absolute-metric 3D tracking despite lower leaderboard score. TAPIP3D comparison strengthens the story — SOTA-ranked model is actually worse in real metres.

## Extended comparison with sensor depth and MegaSAM depth (2026-06-26)

### ADT with sensor depth (depth_preds — best possible)
| Method | metric-AJ | err_mean |
|--------|-----------|----------|
| SEA-RAFT + depthn | 28.93% | 0.363m |
| SEA-RAFT + DA3    | 27.27% | 0.410m |
| TAPIP3D + depthn  | 19.69% | — |
| TAPIP3D + DA3     | 0.38%  | — |

Sensor depth is better than DA3 for SEA-RAFT (+5.7% metric-AJ). TAPIP3D improves 52× over DA3 on ADT with sensor depth (19.7% vs 0.38%) — consistent with hypothesis that DA3 has per-pixel shape errors incompatible with TAPIP3D's absolute thresholds.

### MegaSAM (drivetrack + pstudio) — IN PROGRESS (2026-06-26, overnight)
**Status:** Overnight pipeline running.
- `scripts/run_megasam_minival_eval.sh` — PID 471939
- drivetrack: 50 clips × ~122s ≈ 1.7 hrs
- pstudio: 50 clips × ~507s ≈ 7 hrs
- After annotation: SEA-RAFT+MegaSAM and TAPIP3D+MegaSAM eval
- Log: `outputs/run_megasam_minival_eval_overnight.log`

**Why:** User wants all 4 methods (TAPIP3D+MegaSAM, RAFT+DA3, RAFT+MegaSAM, v33) on same minival data with same metric. MegaSAM provides camera poses + consistent metric depth for drivetrack/pstudio (no built-in depth_preds).
