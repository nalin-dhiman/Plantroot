#!/usr/bin/env python3
"""Diagnose a ROOT-FPT source checkout and run a small deterministic smoke test."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check imports and interpreter only; skip the simulation smoke test.",
    )
    arguments = parser.parse_args()

    print("ROOT-FPT installation doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Virtual environment: {'yes' if sys.prefix != sys.base_prefix else 'no'}")
    if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
        print("FAIL: ROOT-FPT supports Python 3.11 and 3.12.")
        return 1

    try:
        import streamlit

        from rootfpt import __version__
        from rootfpt.explorer import result_signature, run_experiment
    except ModuleNotFoundError as error:
        print(f"FAIL: missing module {error.name!r}.")
        print('Run: python -m pip install -e ".[app]"')
        return 1

    print(f"ROOT-FPT: {__version__}")
    print(f"Streamlit: {streamlit.__version__}")
    print(f"Source: {SOURCE_ROOT}")
    try:
        print(f"Installed distribution: rootfpt {version('rootfpt')}")
    except PackageNotFoundError:
        print("Warning: running from the source fallback; editable install not detected.")
    if not arguments.quick:
        result = run_experiment(
            "taproot",
            "homogeneous",
            seed=17,
            replicate=0,
            duration_days=0.25,
            dt_days=0.04,
            max_tips=40,
        )
        print(f"Smoke-test segments: {len(result.architecture.segments)}")
        print(f"Smoke-test signature: {result_signature(result)[:16]}")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
