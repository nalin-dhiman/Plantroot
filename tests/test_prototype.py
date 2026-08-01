from __future__ import annotations

import math
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from root_fpt_prototype import ModelConfig, MoistureField, simulate


def test_moisture_is_bounded() -> None:
    cfg = ModelConfig()
    f = MoistureField(cfg)
    for x in (-1.0, 0.0, 1.0):
        for z in (0.0, 0.5, 1.5, 2.1):
            value, gx, gz = f.value_and_gradient(x, z, 0.0)
            assert 0.0 <= value <= 1.0
            assert math.isfinite(gx)
            assert math.isfinite(gz)


def test_simulation_respects_budget() -> None:
    cfg = ModelConfig(seed=3, duration=5.0, construction_budget=0.8)
    result = simulate(cfg)
    # One discrete step can create a tiny numerical overshoot only through the
    # branch event charge; the model checks branch affordability explicitly.
    assert result.length_used <= cfg.construction_budget + 1e-9
    assert len(result.tips) >= 1


def test_simulation_is_reproducible() -> None:
    cfg1 = ModelConfig(seed=23, duration=4.0)
    cfg2 = ModelConfig(seed=23, duration=4.0)
    r1 = simulate(cfg1)
    r2 = simulate(cfg2)
    assert r1.branch_events == r2.branch_events
    assert len(r1.tips) == len(r2.tips)
    assert abs(r1.length_used - r2.length_used) < 1e-12
    assert r1.hit == r2.hit
