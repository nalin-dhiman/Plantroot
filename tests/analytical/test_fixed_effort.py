from __future__ import annotations

import math

from rootfpt.experiments.fixed_effort import canonical_schedules, verify_fixed_effort


def test_all_canonical_schedules_have_exact_equal_effort() -> None:
    schedules = canonical_schedules()
    assert len(schedules) == 5
    assert {schedule.integrated_effort for schedule in schedules} == {12.0}
    assert all(
        math.isclose(schedule.scaled_to(7.3).integrated_effort, 7.3)
        for schedule in schedules
    )


def test_rescaled_progressive_schedule_has_byte_stable_effort() -> None:
    schedule = canonical_schedules()[2].scaled_to(2.0)
    assert schedule.integrated_effort == 2.0


def test_fixed_effort_monte_carlo_collapses_with_declared_smoke_tolerance() -> None:
    table = verify_fixed_effort(
        efforts=(3.0, 9.0),
        hazard=0.08,
        replicates=4000,
        master_seed=51,
        absolute_tolerance=0.035,
    )
    assert bool(table["passed"].all())
    assert table["absolute_error"].max() < 0.035
