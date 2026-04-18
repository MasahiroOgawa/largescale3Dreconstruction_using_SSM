---
name: Never use pip directly — use uv
description: User explicitly rejects any pip invocation; all Python dependency and env ops must go through uv
type: feedback
---

Never invoke `pip`, `pip install`, `pip install -e .`, `python -m pip`, or `uv pip install`. Use `uv` natively for everything.

**Why:** User reinforced this rule in-session despite it already being in global CLAUDE.md — treat as a hard rule, not a preference. Any pip invocation is a mistake.

**How to apply:**
- Third-party packages: `uv add <pkg>`
- Editable local sources: configure `[tool.uv.sources]` in `pyproject.toml` with `path = ...` + `editable = true`, then `uv add <name>`. Do NOT run `uv pip install -e .`
- Running scripts: `uv run python ...`, `uv run pytest`
- Sync after editing pyproject.toml: `uv sync`
- If docs say "pip install ...", translate to the uv form before presenting.
