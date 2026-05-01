"""Thin CLI entry — the real demo lives in scripts/run_demo.py."""

from __future__ import annotations


def main() -> None:
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_demo.py"
    sys.exit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))
