# Project Rules — largescale3Dreconstruction_using_SSM

These rules are repo-local. They override any global `~/.claude/CLAUDE.md` behavior for this project. All memory files live **in this repo**, never under `~/.claude/`.

## Python / packaging

- **Never invoke `pip` directly.** Not `pip install`, not `pip install -e .`, not `python -m pip`, not `pip3`, not `uv pip install`. All are forbidden.
- **Use `uv` for everything.** The venv is managed by `uv`.
  - Add third-party packages: `uv add <pkg>`
  - Add a local editable source (e.g., `third_party/depth-anything-3`): configure `[tool.uv.sources]` in `pyproject.toml` with `path = "..."` + `editable = true`, then `uv add <name>`
  - Run anything in the venv: `uv run python ...`, `uv run pytest`
  - Sync after manual edits to `pyproject.toml`: `uv sync`
- If an upstream README says "pip install ...", translate it to the uv equivalent before running it.

## Depth-Anything-3 integration

- DA3 is a git submodule at `third_party/depth-anything-3`. Treat it as read-only upstream code — **never edit files inside that directory**. Swap attention modules at runtime via `mamba3_attn.patch.install_mamba3(...)`.
- If DA3 upstream changes and the patch breaks, fix `src/mamba3_attn/patch.py` or `src/mamba3_attn/da3_adapter.py` — never the submodule.

## Memory

- Memory files for this project live at `memory/*.md` under this repo, indexed by `MEMORY.md` at the repo root.
- Global `~/.claude/projects/.../memory/` is off-limits for this project.

See `MEMORY.md` for the current index.

## Sudo

- Never run `sudo` directly. If a command needs sudo, print the exact command and ask the user to run it.

## Code style

- Prefer concise expressions (comprehensions over for+append loops).
- Do not add comments that restate the code; only add comments when the *why* is non-obvious.
