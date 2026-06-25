---
name: project-tapip3d-comparison
description: TAPIP3D vs v33 comparison on TAPVid-3D minival — final results
metadata:
  type: project
---

Comparing TAPIP3D (image-only, DA3 depth) with v33 on the same 150 minival clips.

**Why:** v33 evaluates on minival; TAPIP3D published only full-eval numbers. Running both on minival gives a direct apples-to-apples comparison.

**Both use `scaling="median"` (standard leaderboard protocol) — metrics are directly comparable.**

**Final results (median-scaled 3D-AJ, minival):**

| Subset     | TAPIP3D (DA3) | v33 (SEA-RAFT+DA3+Mamba3) | Winner |
|------------|--------------|--------------------------|--------|
| drivetrack | **6.49%**    | 1.35%                    | TAPIP3D +5.1pp |
| pstudio    | **2.28%**    | 0.83%                    | TAPIP3D +1.5pp |
| adt        | 0.38%        | **0.58%**                | v33 +0.2pp |
| **mean**   | **3.05%**    | 0.92%                    | TAPIP3D +2.1pp |

TAPIP3D is substantially better overall. v33 has a small edge only on ADT.
Note: v33 beats TAPIP3D on ADT in absolute-metric eval too (from [[project-metric3d-evaluation]]).

**Pipeline (all complete):**
1. DA3 depth for all 150 minival clips → `outputs/tapvid3d_da3/<subset>/<clip>/depth.npz`
2. TAPIP3D HDF5 annotations → `outputs/tapip3d_annotations/<subset>_da3_minival/da3/`
3. TAPIP3D eval output → `outputs/tapip3d_eval_minival_20260624-1855/`
4. TAPIP3D metrics JSON → `/home/mas/proj/study/TAPIP3D/outputs/auto_generated/tapip3d_kubric_24frames_384trajs_2026-06-24_18-55-53/metrics_<subset>_da3_minival.json`

**How to apply:** TAPIP3D is a strong baseline. For the paper, v33 needs to be improved or the ADT advantage + absolute-metric advantage must be the focus.
