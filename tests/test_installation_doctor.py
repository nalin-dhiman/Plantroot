from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_installation_doctor_quick_check() -> None:
    repository = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, "scripts/doctor.py", "--quick"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "Status: PASS" in process.stdout
