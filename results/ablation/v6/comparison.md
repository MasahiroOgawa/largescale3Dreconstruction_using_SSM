# TAPVid-3D comparison

## 3D-AJ per subset (higher is better)

| method | aria | drivetrack | pstudio | mean |
|---|---|---|---|---|
| BootsTAPIR + ZoeDepth | 0.123 | 0.075 | 0.213 | 0.137 |
| SpatialTracker | 0.137 | 0.094 | 0.225 | 0.152 |
| CoTracker3D-online | 0.146 | 0.107 | 0.213 | 0.155 |
| CoTracker3D-offline | 0.163 | 0.135 | 0.222 | 0.173 |
| DELTA | 0.176 | 0.130 | 0.244 | 0.183 |
| **Mamba-3 tracker (this run)** | **nan** | **0.001** | **0.008** | **0.005** |

## Per-subset detail of this run

| subset | 3D-AJ | APD3D | OA |
|---|---|---|---|
| adt | nan | nan | nan |
| drivetrack | 0.0013 | 0.0013 | 0.7890 |
| pstudio | 0.0077 | 0.0077 | 0.7581 |

Baseline numbers from configs/tapvid3d_baselines.yaml (TAPVid-3D paper + each method's released numbers).
