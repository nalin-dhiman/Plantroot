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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also require an installed rootfpt distribution matching the source version.",
    )
    arguments = parser.parse_args()

    print("ROOT-FPT installation doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    in_virtual_environment = sys.prefix != sys.base_prefix
    print(f"Virtual environment: {'yes' if in_virtual_environment else 'no'}")
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
    installed_distribution = None
    try:
        installed_distribution = version("rootfpt")
        print(f"Installed distribution: rootfpt {installed_distribution}")
    except PackageNotFoundError:
        print("Warning: running from the source fallback; editable install not detected.")
    if arguments.strict and installed_distribution is None:
        print('FAIL: strict mode requires: python -m pip install -e ".[app]"')
        return 1
    if arguments.strict and installed_distribution != __version__:
        print(
            "FAIL: installed distribution and source checkout have different versions "
            f"({installed_distribution!r} != {__version__!r})."
        )
        print('Run: python -m pip install -e ".[app]"')
        return 1
    for required_path in (REPOSITORY_ROOT / "streamlit_app.py", REPOSITORY_ROOT / "app/main.py"):
        if not required_path.is_file():
            print(f"FAIL: missing application file {required_path.relative_to(REPOSITORY_ROOT)}")
            return 1
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
