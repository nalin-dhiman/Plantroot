#!/usr/bin/env python3
"""Launch the ROOT-FPT Explorer with the same Python used for installation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        import streamlit  # noqa: F401
    except ModuleNotFoundError:
        print("Streamlit is not installed for this Python interpreter.")
        print('Run: python -m pip install -e ".[app]"')
        return 1
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(REPOSITORY_ROOT / "streamlit_app.py"),
        *sys.argv[1:],
    ]
    return subprocess.call(command, cwd=REPOSITORY_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
