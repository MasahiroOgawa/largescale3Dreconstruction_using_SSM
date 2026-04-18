---
name: DA3 submodule is read-only
description: Depth-Anything-3 lives in third_party/ as a git submodule and must never be edited in-place
type: feedback
---

`third_party/depth-anything-3/` is a pristine submodule. Never edit any file under it.

**Why:** Upstream drift and merge conflicts are avoided by keeping DA3 unmodified. Our swap is applied at runtime via a patcher, not by forking.

**How to apply:**
- Swap self-attention via `ssm3d.patch.install_mamba3(net)` which walks `net.backbone.blocks` and replaces `block.attn`.
- New functionality that would otherwise be an edit to DA3 goes into `src/ssm3d/` as an adapter, subclass, or patch.
- If the patcher breaks on an upstream change, fix the patcher — not the submodule.
