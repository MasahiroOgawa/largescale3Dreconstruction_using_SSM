---
name: output-dir-naming-evaltitle-datetime
description: Eval / render / artifact output directories must be named `<eval_title>_<datetime>` (e.g. `v11_step22500_20260521-1810`). No bare `<eval_title>` or `<eval_title>_stepN` without a timestamp suffix.
metadata:
  type: feedback
---

When creating output directories for eval runs, render runs, training-curve
plots, or any artifact dump produced from a checkpoint, name the directory:

```
<eval_title>_<datetime>
```

where:
- `eval_title` summarises *what* the run is (e.g. `v11_step22500`,
  `v10_eval_pstudio_drivetrack`, `compare_v8_v10_viz`).
- `datetime` is a sortable timestamp, format `YYYYMMDD-HHMM` (local time).

Examples that match the rule:
- `outputs/eval_tracker/v11_step22500_20260521-1810/viz/`
- `results/ablation/v10_eval_20260521-0807/`
- `outputs/plots/v11_loss_curves_20260521-1100/`

Examples that **violate** the rule (do not do these):
- `outputs/eval_tracker/v11_step22500/`            (no datetime)
- `results/ablation/v11/`                          (no datetime)
- `outputs/eval_tracker/v11/viz/`                  (no datetime)

**Why:** running the same script twice on the same checkpoint at different
times currently overwrites the previous output. With a datetime suffix
each render/eval is a separately addressable artifact, easy to diff
visually, and never silently clobbered. User asked for this on 2026-05-21
after several ad-hoc render dirs accumulated indistinguishably.

**Scope:**
- Applies to `outputs/eval_tracker/...`, `outputs/plots/...`,
  `results/ablation/...`, and any other artifact dir created from a
  checkpoint.
- Does NOT apply to `outputs/runs/<variant>/...` (those are the training
  runs themselves; their checkpoint files inside ARE step-named, and
  resume-from-latest depends on the canonical dir name).
- Does NOT apply to `configs/` (config files are versioned by name).
