#!/usr/bin/env python3
"""Compare all synthetic presets in one shared patchy-soil realization."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows"))

import yaml  # noqa: E402
from run_root_soil_atlas import CONFIG, simulate_one  # noqa: E402

config = yaml.safe_load(CONFIG.read_text())
for name in config["root_regimes"]:
    _, _, metrics, _, seeds = simulate_one(name, "patchy_matern", 7, config)
    print(
        f"{name:28s} environment_seed={seeds['environment_seed']} "
        f"depth={metrics['maximum_depth']:.3f} length={metrics['total_root_length']:.3f}"
    )
