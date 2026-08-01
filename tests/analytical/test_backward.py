from __future__ import annotations

import numpy as np

from rootfpt.experiments.backward import (
    no_branch_half_line_solution,
    solve_backward_1d,
)


def test_backward_no_branch_matches_half_line_solution() -> None:
    solution = solve_backward_1d(
        diffusion=0.08,
        branch_rate=0.0,
        mortality_rate=0.0,
        domain_size=8.0,
        deadline=2.0,
        nx=801,
    )
    analytical = no_branch_half_line_solution(solution.x, 0.08, 2.0)
    assert np.max(np.abs(solution.probability[:401] - analytical[:401])) < 0.012
    assert solution.bound_residual <= 1e-10


def test_backward_expected_monotonicities() -> None:
    common = {
        "diffusion": 0.08,
        "domain_size": 4.0,
        "deadline": 4.0,
        "nx": 201,
    }
    base = solve_backward_1d(branch_rate=0.06, mortality_rate=0.03, **common)
    more_branch = solve_backward_1d(branch_rate=0.12, mortality_rate=0.03, **common)
    more_death = solve_backward_1d(branch_rate=0.06, mortality_rate=0.08, **common)
    assert np.all((base.probability >= 0.0) & (base.probability <= 1.0))
    assert np.all(more_branch.probability + 1e-10 >= base.probability)
    assert np.all(more_death.probability <= base.probability + 1e-10)

