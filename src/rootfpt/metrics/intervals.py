"""Frequentist uncertainty intervals used by Monte Carlo baselines."""

from __future__ import annotations

import math


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (proportion + z2 / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)
