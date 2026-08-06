---
name: project-repo-split
description: Monorepo split into visionMamba3 (pure SSM library) and vmamba3-3Dpointtracker (tracker app); package renamed to visionmamba3
metadata:
  type: project
---

The monorepo `largescale3Dreconstruction_using_SSM` has been split into two repos:

**`/home/mas/proj/study/visionMamba3`** (GitHub: MasahiroOgawa/visionMamba3)
- Package: `visionmamba3` (was `mamba3_attn.mamba3`)
- Contents: `src/visionmamba3/` — pure Mamba-3 SSD attention modules only
- third_party: `mamba-ssm` as absolute symlink → `/home/mas/proj/study/mamba-ssm`
- No DA3 code (da3_adapter, patch, train, eval stay in old monorepo)

**`/home/mas/proj/study/vmamba3-3Dpointtracker`** (GitHub: MasahiroOgawa/vmamba3-3Dpointtracker)
- Packages: `mamba3_tracker`, `searaft_flow`
- third_party/visionMamba3: proper git submodule (version-pinned)
- All other third_party: absolute symlinks to `~/proj/study/{name}`
- Import paths: `from visionmamba3.cross_attention import Mamba3CrossAttention` (not `mamba3_attn.mamba3`)

**Why:** The old monorepo mixed the reusable SSM library with the application tracker.
**How to apply:** Any new development of Mamba-3 attention goes in visionMamba3; tracker work goes in vmamba3-3Dpointtracker. The old monorepo (`largescale3Dreconstruction_using_SSM`) remains as archive for DA3 training/eval code.

**third_party symlink layout at ~/proj/study/:**
- `mamba-ssm` → `mamba` (was cloned as `mamba` by user)
- `depth-anything-3` → `largescale3Dreconstruction_using_SSM/third_party/depth-anything-3`
- `SEA-RAFT` → `largescale3Dreconstruction_using_SSM/third_party/SEA-RAFT`
- `SpaTrackerV2`, `TrackCraft3R` — proper independent clones
