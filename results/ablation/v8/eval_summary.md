# Mamba-3 Tracker — TAPVid-3D evaluation (v8)

Checkpoint: `outputs/runs/mamba3_tracker_v8/ckpt_30000.pt` (step 30000)

## Per-subset means

| subset | 3D-AJ | APD3D | OA | clips | frames per clip |
|---|---|---|---|---|---|
| pstudio    | 0.0961 | 0.0961 | 0.7581 | ~150 | full |
| drivetrack | 0.0020 | 0.0020 | 0.7890 | ~50  | full |
| adt        | 0.0768 | 0.0768 | 0.5536 | 200  | first 64 only (see caveat) |
| **mean (pstudio + drivetrack)** | **0.0490** | **0.0490** | **0.7735** | | |
| **mean (all 3 subsets)**        | **0.0583** | **0.0583** | **0.7002** | | |

Same ADT max-frames=64 caveat as v7 (encoder OOMs on full ADT clips).
Pstudio and drivetrack are full-clip.
