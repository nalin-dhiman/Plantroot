from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solve_backward_1d import solve
from verify_fixed_effort_law import schedules


def test_fixed_effort_schedules_are_matched() -> None:
    effort = 7.3
    dt = 0.02
    for n_t in schedules(effort, dt).values():
        assert math.isclose(float(n_t.sum() * dt), effort, rel_tol=1e-12, abs_tol=1e-12)


def test_backward_solution_is_probability_and_decreases_with_distance() -> None:
    x, p = solve(D=0.08, lam=0.08, mu=0.04, L=4.0, deadline=8.0, nx=201)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert p[0] == 1.0
    # Small numerical fluctuations are tolerated, but the solution should be
    # overwhelmingly non-increasing away from the target.
    assert np.mean(np.diff(p) <= 1e-8) > 0.98
