---
name: project-tapip3d-comparison
description: Status of TAPIP3D vs v33 comparison on TAPVid-3D minival
metadata:
  type: project
---

Comparing TAPIP3D (image-only, DA3 depth) with v33 on the same 150 minival clips.

**Why:** v33 evaluates on minival; TAPIP3D published only full-eval numbers. Running both on minival gives a direct apples-to-apples comparison.

**Pipeline status:**
1. DA3 depth for drivetrack (50 clips): done → `outputs/tapvid3d_da3/drivetrack/`
2. DA3 depth for pstudio (50 clips): done → `outputs/tapvid3d_da3/pstudio/`
3. DA3 depth for adt (50 clips, 300 frames each): **running** (PID 2560696, ~12 min)
4. TAPIP3D HDF5 annotation generation: not yet run
5. TAPIP3D eval (minival 3-subset): not yet run

**Key paths:**
- DA3 depth outputs: `outputs/tapvid3d_da3/<subset>/<clip>/depth.npz`
- TAPIP3D annotations: `outputs/tapip3d_annotations/<subset>_da3_minival/da3/<seq_id>.h5`
- Annotation script: `/home/mas/proj/study/TAPIP3D/scripts/make_da3_annotations_minival.py`
- Eval config: `/home/mas/proj/study/TAPIP3D/configs/tapip3d_da3_minival_eval.yaml`

**Run commands (after DA3 finishes):**

Step A – package annotations (in TAPIP3D venv):
```bash
TAPIP3D=/home/mas/proj/study/TAPIP3D
VENV=$TAPIP3D/.venv
CU13=$VENV/lib/python3.11/site-packages/nvidia/cu13
LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/torch/lib:$CU13/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$TAPIP3D:$PYTHONPATH \
$VENV/bin/python $TAPIP3D/scripts/make_da3_annotations_minival.py --subsets drivetrack pstudio adt
```

Step B – run TAPIP3D eval (single GPU, ~1-2h):
```bash
cd /home/mas/proj/study/TAPIP3D
TAPIP3D_OUT=outputs/tapip3d_eval_minival_$(date +%Y%m%d-%H%M)
VENV=$PWD/.venv
CU13=$VENV/lib/python3.11/site-packages/nvidia/cu13
LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/torch/lib:$CU13/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$PWD:$PYTHONPATH \
$VENV/bin/accelerate launch \
  --num_processes 1 \
  train_eval.py --config-name tapip3d_da3_minival_eval \
  '~test_datasets.kubric_24frames_384trajs_200samples' \
  +train.eval_only=true \
  +model.eval_mode=raw \
  train.checkpoint=checkpoints/tapip3d_final.pth \
  +train.visualize_with_rerun=false \
  train.mixed_precision=bf16 \
  output_dir=$TAPIP3D_OUT
```

**TAPIP3D pointops2 build:** Built for sm_89 (RTX 4080 Laptop).
Requires LD_LIBRARY_PATH to include torch/lib and cu13/lib at runtime.

**v33 minival results (for comparison):**
- drivetrack: 3D-AJ mean 1.35%
- pstudio: 3D-AJ mean 0.83%  
- adt: 3D-AJ mean 0.58%
- mean: 3D-AJ ~0.92%

**How to apply:** Once TAPIP3D eval finishes, extract `average_jaccard` from
`metrics_drivetrack_da3_minival.json` etc., compare with v33 numbers above.
