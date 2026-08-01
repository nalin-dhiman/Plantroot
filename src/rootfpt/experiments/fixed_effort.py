"""Exact fixed-effort null-law schedules and Monte Carlo verification."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rootfpt.metrics.intervals import wilson_interval


@dataclass(frozen=True)
class EffortSchedule:
    """A piecewise-constant active-tip schedule."""

    name: str
    intervals: tuple[tuple[float, int], ...]

    @property
    def integrated_effort(self) -> float:
        # ``math.fsum`` makes the equal-effort invariant stable across supported
        # Python versions after schedules are rescaled.
        return math.fsum(
            duration * active_tips for duration, active_tips in self.intervals
        )

    def scaled_to(self, effort: float) -> EffortSchedule:
        if effort <= 0:
            raise ValueError("effort must be positive")
        current = self.integrated_effort
        if current <= 0:
            raise ValueError("schedule must have positive effort")
        factor = effort / current
        return EffortSchedule(
            self.name,
            tuple((duration * factor, active_tips) for duration, active_tips in self.intervals),
        )


def canonical_schedules() -> tuple[EffortSchedule, ...]:
    """Return five literal schedules, each with baseline effort 12 tip-time units."""
    return (
        EffortSchedule("one_tip_full_duration", ((12.0, 1),)),
        EffortSchedule("simultaneous_short_lived", ((3.0, 4),)),
        EffortSchedule("progressive_branching", ((3.0, 1), (2.0, 2), (1.0, 5))),
        EffortSchedule("late_burst", ((9.0, 0), (3.0, 4))),
        EffortSchedule("early_burst", ((3.0, 4), (9.0, 0))),
    )


def simulate_schedule(
    schedule: EffortSchedule,
    hazard: float,
    replicates: int,
    rng: np.random.Generator,
) -> int:
    """Simulate whether at least one hit occurs in each replicate."""
    if hazard < 0:
        raise ValueError("hazard must be nonnegative")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    alive = np.ones(replicates, dtype=bool)
    for duration, active_tips in schedule.intervals:
        if duration < 0 or active_tips < 0:
            raise ValueError("schedule duration and tip count must be nonnegative")
        if active_tips == 0:
            continue
        interval_hit_probability = -math.expm1(-hazard * active_tips * duration)
        draws = rng.random(replicates)
        alive &= draws >= interval_hit_probability
    return int((~alive).sum())


def verify_fixed_effort(
    *,
    efforts: tuple[float, ...],
    hazard: float,
    replicates: int,
    master_seed: int,
    absolute_tolerance: float,
) -> pd.DataFrame:
    """Evaluate all schedules and declare collapse using CI or MC tolerance."""
    root = np.random.SeedSequence(master_seed)
    jobs = [(effort, template) for effort in efforts for template in canonical_schedules()]
    sequences = root.spawn(len(jobs))
    rows: list[dict[str, float | int | str | bool]] = []
    for (effort, template), sequence in zip(jobs, sequences, strict=True):
        schedule = template.scaled_to(effort)
        successes = simulate_schedule(
            schedule,
            hazard,
            replicates,
            np.random.default_rng(sequence),
        )
        estimate = successes / replicates
        theory = -math.expm1(-hazard * effort)
        low, high = wilson_interval(successes, replicates)
        absolute_error = abs(estimate - theory)
        relative_error = absolute_error / theory if theory > 0 else 0.0
        passed = (low <= theory <= high) or (absolute_error <= absolute_tolerance)
        rows.append(
            {
                "effort": effort,
                "schedule": schedule.name,
                "integrated_effort": schedule.integrated_effort,
                "hazard": hazard,
                "replicates": replicates,
                "successes": successes,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "theory": theory,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "absolute_tolerance": absolute_tolerance,
                "passed": passed,
                "seed_spawn_key": ".".join(str(item) for item in sequence.spawn_key),
            }
        )
    return pd.DataFrame(rows)
