"""Machine-readable run provenance."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def git_commit(root: Path) -> str:
    """Return the current Git commit or an explicit sentinel."""
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "UNINITIALIZED"


def dependency_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_manifest(
    *,
    root: Path,
    configuration_hash: str,
    seed_manifest: dict[str, Any],
    replicates: int,
    tolerances: dict[str, float],
    wall_time_seconds: float,
    failed_runs: int = 0,
    excluded_runs: int = 0,
    exclusion_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build the minimum provenance record required for an eligible result."""
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "dependencies": dependency_versions(
            ("rootfpt", "numpy", "scipy", "pandas", "matplotlib", "PyYAML", "pyarrow")
        ),
        "configuration_hash": configuration_hash,
        "seed_hierarchy": seed_manifest,
        "replicates": replicates,
        "solver_tolerances": tolerances,
        "wall_time_seconds": wall_time_seconds,
        "failed_run_count": failed_runs,
        "excluded_run_count": excluded_runs,
        "exclusion_reasons": exclusion_reasons or [],
    }
