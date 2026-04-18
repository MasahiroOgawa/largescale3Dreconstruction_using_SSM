---
name: Memory lives in repo, not ~/.claude/
description: For this project, all memory files are repo-local under memory/, indexed by MEMORY.md at the repo root
type: feedback
---

All memory for this project is stored in `memory/*.md` at the repo root, indexed by `MEMORY.md`. Never write to `~/.claude/projects/.../memory/`.

**Why:** `~/.claude/` is per-user-machine and doesn't travel with the repo. User wants memory discoverable by collaborators cloning the repo, and consistent across machines.

**How to apply:**
- New memory → create `memory/<name>.md` in the repo, add a line to `MEMORY.md`.
- When saving something that would normally go to the auto-memory path, save to the repo instead.
- If you find stray files under `~/.claude/projects/.../memory/` for this project, delete them.
