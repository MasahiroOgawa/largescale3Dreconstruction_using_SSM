---
name: project_metric3d_evaluation
description: Absolute-metric TAPVid-3D eval reverses the v33 verdict — v33 beats SEA-RAFT+DA3 in real metres; leaderboard median scaling hid it
metadata:
  type: project
---

The TAPVid-3D headline metric is scale-invariant twice over: per clip it (1)
median-rescales predicted depth to GT and (2) uses depth-relative pixel
thresholds. For metric 3D reconstruction (this project's goal) both discard the
quantity we care about. We added an **absolute-metric** evaluation path:
`compute_clip_metrics_absolute` in `src/mamba3_tracker/eval/tapvid3d_eval.py`
calls the official vendored metric with `scaling="none"` +
`use_fixed_metric_threshold=True` (fixed-metre thresholds 1cm..2.56m) →
metric-AJ / metric-APD3D, plus mean/median real 3D error in metres. Runner:
`scripts/eval_metric3d.py` (full minival, `--method searaft|v33`); roll-up:
`scripts/compare_metric3d.py`.

**Full 150-clip minival result (verified; median-AJ reproduces prior eval exactly):**

| | median-AJ | metric-AJ | metric-APD3D | err mean/med (m) |
|---|---|---|---|---|
| SEA-RAFT+DA3 | **0.052** | 0.147 | 0.254 | 4.29 / 4.01 |
| v33 | 0.032 | **0.180** | **0.306** | **2.67 / 2.13** |

**The verdict reverses under the metric that matters.** v33 loses on the
median-scaled leaderboard metric (3.2% vs 5.2%) but WINS every absolute-metric
measure overall: metric-AJ +22%, metric-APD3D +20%, mean error −38% (median
−47%). The gain is concentrated where DA3's metric scale is most wrong:
drivetrack (far-field ~20m) metric-AJ 0.005→0.082 (16×), error 11.70→6.89 m;
adt (indoor ~1m, DA3 already accurate) is a tie — the learned depth corrector
helps when needed, no harm otherwise. Both run ~12.5 fps, 0 failures.

**Why:** median scaling grants every method a free per-clip global-scale fix, so
it cannot see that DA3's raw metric depth is ~40% scale-biased and that v33
corrects it. v33's raw error ≈ its median-scaled error → its output is already
near the correct absolute scale.

**How to apply / paper decision gate.** The "v33 is a failure" conclusion in
[[project_v33_depth_refined_tracker]] is metric-dependent and must be qualified:
v33 is the better model for absolute-metric 3D tracking. A paper framing is
viable ("median scaling hides metric-depth quality; a tiny learned depth
corrector gives SoTA-grade absolute-metric tracking") BUT requires a fair
baseline: **no published absolute-metric SOTA numbers exist** (the whole
leaderboard is median-scaled) and most SOTA trackers aren't metric-depth, so we
must run a metric-capable baseline ourselves (e.g. BootsTAPIR+ZoeDepth) in
absolute metric. That run is the agreed next decision gate, now justified since
the v33 metric win is solid at full scale. See
[[project_searaft_flow_baseline]], [[project_v33_depth_refined_tracker]].
