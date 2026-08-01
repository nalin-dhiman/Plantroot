"""Exact identities and declared approximations for branching-search theory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.special import ndtr


def poisson_success(exposure: float) -> float:
    """Return the Poisson void-complement probability ``1-exp(-exposure)``."""
    if exposure < 0:
        raise ValueError("exposure must be nonnegative")
    return -math.expm1(-exposure)


def marginal_probability(base_exposure: float, new_exposure: float) -> float:
    """Exact success-probability gain from a measurable set increment."""
    if base_exposure < 0 or new_exposure < 0:
        raise ValueError("exposures must be nonnegative")
    return math.exp(-base_exposure) * (-math.expm1(-new_exposure))


def marginal_utility(
    base_exposure: float,
    new_exposure: float,
    incremental_cost: float,
) -> float:
    """Exact change in ``J=1-exp(-W)-C``."""
    if incremental_cost < 0:
        raise ValueError("incremental cost must be nonnegative")
    return marginal_probability(base_exposure, new_exposure) - incremental_cost


@dataclass(frozen=True)
class BranchIncrement:
    """One branch candidate evaluated in a fixed admissible order."""

    new_exposure: float
    incremental_cost: float

    def __post_init__(self) -> None:
        if self.new_exposure < 0 or self.incremental_cost < 0:
            raise ValueError("branch increments must be nonnegative")


def classify_branch_sequence(
    initial_exposure: float,
    increments: list[BranchIncrement],
) -> tuple[Literal["no_branching", "boundary", "interior"], int, list[float]]:
    """Classify the best prefix of a finite candidate sequence.

    The classification is exact for the declared order. If marginal exposure is
    non-increasing and marginal cost non-decreasing, its signs can cross at most
    once, giving the corresponding global prefix optimum.
    """
    if initial_exposure < 0:
        raise ValueError("initial_exposure must be nonnegative")
    exposure = initial_exposure
    objective = poisson_success(exposure)
    objectives = [objective]
    for item in increments:
        exposure += item.new_exposure
        objective += marginal_utility(
            exposure - item.new_exposure,
            item.new_exposure,
            item.incremental_cost,
        )
        objectives.append(objective)
    optimum = max(range(len(objectives)), key=objectives.__getitem__)
    if optimum == 0:
        regime: Literal["no_branching", "boundary", "interior"] = "no_branching"
    elif optimum == len(increments):
        regime = "boundary"
    else:
        regime = "interior"
    return regime, optimum, objectives


def functional_exposure(
    deployment_time: float,
    deadline: float,
    hazard: float,
    transport_delay: float,
    survival_rate: float,
) -> float:
    """Exact constant-hazard exposure with deterministic delay and survival.

    Patch survival is ``exp(-survival_rate * transport_delay)``. Encounters after
    ``deadline - transport_delay`` cannot become functional before the deadline.
    """
    values = (deployment_time, deadline, hazard, transport_delay, survival_rate)
    if any(value < 0 for value in values):
        raise ValueError("times, hazard and survival_rate must be nonnegative")
    available = max(0.0, deadline - transport_delay - deployment_time)
    return hazard * available * math.exp(-survival_rate * transport_delay)


def folded_normal_laplace(mean: float, sigma: float, scale: float) -> float:
    """Return ``E[exp(-|Y|/scale)]`` for ``Y ~ Normal(mean, sigma**2)``."""
    if sigma < 0 or scale <= 0:
        raise ValueError("sigma must be nonnegative and scale positive")
    if sigma == 0:
        return math.exp(-abs(mean) / scale)
    a = 1.0 / scale
    common = math.exp(0.5 * (a * sigma) ** 2)
    return common * (
        math.exp(-a * mean) * ndtr(mean / sigma - a * sigma)
        + math.exp(a * mean) * ndtr(-mean / sigma - a * sigma)
    )


def _outside_interval_probability(mean: float, sigma: float, radius: float) -> float:
    if sigma == 0:
        return float(abs(mean) > radius)
    return float(ndtr((-radius - mean) / sigma) + ndtr((mean - radius) / sigma))


def small_angle_decorrelation(
    *,
    branch_spacing: float,
    persistence_length: float,
    correlation_length: float,
    search_radius: float,
    initial_angle: float = 0.0,
) -> dict[str, float]:
    """Derived daughter-decorrelation bounds in the angular-diffusion model.

    If each daughter orientation obeys ``dtheta=sqrt(2/l_p)dB``, their relative
    transverse displacement at arclength ``s`` is Gaussian with mean
    ``initial_angle*s`` and variance ``4*s**3/(3*l_p)``. A new-exposure screen
    requires separation above ``2*r``; environmental independence is represented
    by ``1-exp(-|Y|/xi)``. The Fréchet bounds avoid the unjustified product used
    in the Phase-2 heuristic.
    """
    if min(branch_spacing, persistence_length, correlation_length) <= 0:
        raise ValueError("length scales must be positive")
    if search_radius < 0:
        raise ValueError("search_radius must be nonnegative")
    s = branch_spacing
    mean = initial_angle * s
    variance = 4.0 * s**3 / (3.0 * persistence_length)
    sigma = math.sqrt(variance)
    tube_distinct = _outside_interval_probability(mean, sigma, 2.0 * search_radius)
    environment_distinct = 1.0 - folded_normal_laplace(
        mean,
        sigma,
        correlation_length,
    )
    lower = max(0.0, tube_distinct + environment_distinct - 1.0)
    upper = min(tube_distinct, environment_distinct)
    return {
        "mean_separation": mean,
        "separation_sd": sigma,
        "tube_distinct_probability": tube_distinct,
        "environment_distinct_factor": environment_distinct,
        "joint_lower_bound": lower,
        "joint_upper_bound": upper,
    }
