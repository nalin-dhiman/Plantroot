#!/usr/bin/env python3
"""Grow one synthetic dimorphic-preset root in one uniform soil."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows"))

import yaml  # noqa: E402
from run_root_soil_atlas import CONFIG, simulate_one  # noqa: E402

config = yaml.safe_load(CONFIG.read_text())
architecture, _, metrics, rld, seeds = simulate_one("dimorphic", "homogeneous", 0, config)
print(f"segments={len(architecture.segments)} seeds={seeds}")
print(
    {
        key: metrics[key]
        for key in ("maximum_depth", "horizontal_spread", "total_root_length", "branch_count")
    }
)
print("root-length density:", rld.tolist())
