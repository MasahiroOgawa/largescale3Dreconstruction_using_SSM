---
name: output-dir-naming-flat-track-version-datetime
description: Single flat output directory per training run — `outputs/track_v<version>_<YYYYMMDD-HHMM>/` holds checkpoints, logs, eval JSONs, render MP4s, and plots together. No separate `outputs/runs/` vs `outputs/eval_tracker/` split.
metadata:
  type: feedback
---

## Convention

Each training run gets **one** top-level directory:

```
outputs/track_v<version>_<YYYYMMDD-HHMM>/
    ├─ cfg.json                         resolved config + launch args snapshot
    ├─ train.log                        stdout of the training process
    ├─ loss_history.json                per-step + per-val loss rows
    ├─ motion_history.json              v13+: [{step, pstudio_ratio, drivetrack_ratio, ...}, ...]
    ├─ ckpt_500.pt, ckpt_1000.pt, ...   training checkpoints
    ├─ ckpt_<final>.pt
    ├─ eval/                            TAPVid-3D eval results (per-subset JSONs)
    │     ├─ pstudio.json
    │     └─ drivetrack.json
    ├─ viz/                             rendered tracking MP4s
    │     ├─ pstudio_basketball_1.mp4
    │     └─ drivetrack_<clip_id>.mp4
    └─ plots/                           training-curve, motion-ratio plots
          ├─ training_curve.png
          └─ motion_ratio.png
```

`<version>` = `v13`, `v14`, etc. `<YYYYMMDD-HHMM>` set at first launch.

## Why one dir

- Resume: pass the same `--out-dir outputs/track_v13_<datetime>` and the train
  script auto-resumes from the latest `ckpt_*.pt` inside.
- Co-located artifacts: easy diff across runs (`diff -r outputs/track_v13_… outputs/track_v14_…`).
- One commit message references one path.
- The user explicitly requested this on 2026-05-22 over the previous split
  (`outputs/runs/...` vs `outputs/eval_tracker/...`).

## Anti-patterns (do NOT do)

- `outputs/eval_tracker/v13_<…>/` — split eval/render away from training.
- `outputs/runs/mamba3_tracker_v13/` — buried under a fixed `mamba3_tracker_*`
  prefix that we used for v6–v12. New runs go straight under `outputs/`.
- `outputs/track_v13/` without a datetime — risks silent overwrite on re-launches.
- `outputs/track_v13_<datetime>/runs/`, `outputs/track_v13_<datetime>/eval/v2/`,
  or any deeper version nesting. One dir per run, period.

## Scope

- Applies to **v13 and later**.
- v6–v12 stay where they are: `outputs/runs/mamba3_tracker_v<6..12>/` and
  `outputs/eval_tracker/...`. No retroactive move.
- The launch command must include the new path explicitly:
  ```bash
  systemd-run --user --scope --quiet -p MemoryMax=18G ... \
      uv run python scripts/train_mamba3_tracker.py \
      --config configs/v13.yaml \
      --out-dir outputs/track_v13_$(date +%Y%m%d-%H%M)
  ```
