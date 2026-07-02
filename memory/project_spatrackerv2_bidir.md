---
name: project-spatrackerv2-bidir
description: SpatialTrackerV2 bidirectional tracking fix — root cause of 2.7% vs 24.7% AJ discrepancy found and fixed
metadata:
  type: project
---

## Root cause of 9× AJ gap (2.7% paper reports 24.7%)

Three bugs in our forward-only `eval_spatracker_v2.py`:
1. **Missing bidirectional tracking** (primary): TAPVid-3D evaluates ALL visible frames including pre-query. ~47.5% of ADT visible frames are pre-query. Forward-only → zero predictions there. Fix: run model on time-reversed video, fill pre-query frames from backward pass.
2. `fixed_cam=False` for ADT/pstudio — caused VO errors. Fix: always `fixed_cam=True`.
3. `replace_ratio=0.2` instead of paper's `1.0`. Fix: `replace_ratio=1.0`.

**Why:** Paper's official evaluator (`SpaTrackerV2/models/SpaTrackV2/evaluation/core/evaluator.py`) lines 132-159 uses bidir tracking. We discovered this by reading the evaluator code.

**Verification**: 2-clip test showed:
- Clip 1: median error 49.8cm → 30.0cm (40% reduction)
- Clip 2: median error 55.9cm → 21.9cm (61% reduction)

## Implementation

`scripts/eval_spatracker_v2.py` updated with:
- `_run_forward`: `fixed_cam=True`, `replace_ratio=1.0`
- `_run_batched`: batching helper (unchanged behavior)
- `infer_clip`: bidirectional protocol — run forward + time-reversed backward, combine with `fwd_mask = t_arr >= t_queries[None, :]`

## Current inference run (started 2026-07-01 ~19:18 JST)

```
PID: 815148
Log: /tmp/spatracker_bidir_20260701_191759.log
Output: /home/mas/data/tapvid3d_baseline_preds/spatrackerv2_bidir/
Subsets: adt pstudio drivetrack (in that order)
```

Estimated completion: 2026-07-02 ~05:45 JST
- ADT: 50 clips × ~8 min ≈ 6.5h → 01:45
- pstudio: 50 clips × ~3 min ≈ 2.5h → 04:15
- drivetrack: 50 clips × ~2 min ≈ 1.5h → 05:45

## After inference completes

1. Score with eval_metric3d.py:
   ```bash
   uv run python scripts/eval_metric3d.py \
     --method spatrackerv2_bidir \
     --pred-dir /home/mas/data/tapvid3d_baseline_preds/spatrackerv2_bidir \
     --split minival --subsets adt pstudio drivetrack
   ```
   *(Note: eval_metric3d.py may need a `--pred-dir` flag or equivalent; check actual CLI)*

2. Update hardcoded scores in `scripts/plot_6method_comparison.py` lines ~151-152:
   ```python
   spatrackerv2_norm = {"drivetrack": NEW, "pstudio": NEW, "adt": NEW}
   spatrackerv2_abs  = {"drivetrack": NEW, "pstudio": NEW, "adt": NEW}
   ```

3. Re-run `scripts/plot_6method_comparison.py` to update figures.

## Old (wrong) forward-only scores in plot_6method_comparison.py

```python
spatrackerv2_norm = {"drivetrack": 0.014054, "pstudio": 0.023991, "adt": 0.026550}
spatrackerv2_abs  = {"drivetrack": 0.008235, "pstudio": 0.152648, "adt": 0.156285}
```

These will be replaced after bidir scoring.

**How to apply:** Check if inference is done (`ls /home/mas/data/tapvid3d_baseline_preds/spatrackerv2_bidir/drivetrack/ | wc -l` == 50), then score and update figures.
