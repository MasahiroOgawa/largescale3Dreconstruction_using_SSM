# Mamba-3 Tracker — TAPVid-3D evaluation (v7)

Checkpoint: `outputs/runs/mamba3_tracker_v7/ckpt_30000.pt` (step 30000)

## Per-subset means

| subset | 3D-AJ | APD3D | OA | clips | frames per clip |
|---|---|---|---|---|---|
| pstudio    | 0.0769 | 0.0769 | 0.7581 | ~150 | full |
| drivetrack | 0.0104 | 0.0104 | 0.7932 | ~50  | full |
| adt        | 0.0908 | 0.0908 | 0.5536 | 200  | first 64 only (see caveat) |
| **mean (pstudio + drivetrack, apples-to-apples vs v6)** | **0.0437** | **0.0437** | **0.7756** | | |
| **mean (all 3 subsets)**                               | **0.0594** | **0.0594** | **0.7016** | | |

**ADT caveat.** ADT clips are ~300 frames at 512×512 and the full-clip
eval encoder activation OOM'd on the 11.6 GB GPU even with
`torch.no_grad()` + bf16. We capped ADT at `--max-frames 64` to produce
a number; this is *not* directly comparable to published TAPVid-3D
baselines, which evaluate on full clips. The pstudio / drivetrack
numbers ARE full-clip and ARE comparable.

A chunked-encoder eval path is the proper fix; tracked as TODO for v8.
