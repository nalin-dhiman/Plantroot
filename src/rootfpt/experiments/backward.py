"""Diagnostic solver for the reduced nonlinear backward equation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import erfc

from rootfpt.metrics.intervals import wilson_interval


@dataclass(frozen=True)
class BackwardSolution:
    x: np.ndarray
    probability: np.ndarray
    dx: float
    dt: float
    steps: int
    minimum: float
    maximum: float
    bound_residual: float


def solve_backward_1d(
    *,
    diffusion: float,
    branch_rate: float,
    mortality_rate: float,
    domain_size: float,
    deadline: float,
    nx: int = 401,
    dt: float | None = None,
    diffusion_cfl: float = 0.42,
    bound_tolerance: float = 1e-10,
) -> BackwardSolution:
    """Solve p_tau = D p_xx + (lambda-mu)p - lambda p^2.

    The target at x=0 is absorbing for failure/no-hit and therefore has p=1.
    The far boundary is reflecting. The method does not clip probabilities;
    an out-of-bounds state is a numerical failure.
    """
    if diffusion <= 0 or domain_size <= 0 or deadline <= 0:
        raise ValueError("diffusion, domain_size, and deadline must be positive")
    if branch_rate < 0 or mortality_rate < 0:
        raise ValueError("rates must be nonnegative")
    if nx < 21:
        raise ValueError("nx must be at least 21")
    x = np.linspace(0.0, domain_size, nx)
    dx = float(x[1] - x[0])
    maximum_dt = diffusion_cfl * dx * dx / diffusion
    requested_dt = maximum_dt if dt is None else dt
    if requested_dt > maximum_dt * (1.0 + 1e-12):
        raise ValueError(
            f"dt={requested_dt:g} violates explicit diffusion limit {maximum_dt:g}"
        )
    steps = max(1, math.ceil(deadline / requested_dt))
    actual_dt = deadline / steps
    probability = np.zeros(nx, dtype=float)
    probability[0] = 1.0
    minimum = 0.0
    maximum = 1.0
    for _ in range(steps):
        laplacian = np.empty_like(probability)
        laplacian[1:-1] = (
            probability[2:] - 2.0 * probability[1:-1] + probability[:-2]
        ) / dx**2
        laplacian[-1] = 2.0 * (probability[-2] - probability[-1]) / dx**2
        laplacian[0] = 0.0
        interior = probability[1:]
        probability[1:] = interior + actual_dt * (
            diffusion * laplacian[1:]
            + (branch_rate - mortality_rate) * interior
            - branch_rate * interior**2
        )
        probability[0] = 1.0
        minimum = min(minimum, float(probability.min()))
        maximum = max(maximum, float(probability.max()))
        if minimum < -bound_tolerance or maximum > 1.0 + bound_tolerance:
            raise FloatingPointError(
                f"probability bounds violated: min={minimum:g}, max={maximum:g}"
            )
    residual = max(0.0, -minimum, maximum - 1.0)
    return BackwardSolution(
        x=x,
        probability=probability,
        dx=dx,
        dt=actual_dt,
        steps=steps,
        minimum=minimum,
        maximum=maximum,
        bound_residual=residual,
    )


def no_branch_half_line_solution(x: np.ndarray, diffusion: float, deadline: float) -> np.ndarray:
    """Analytical no-mortality half-line hitting probability."""
    return erfc(x / (2.0 * math.sqrt(diffusion * deadline)))


def convergence_study(
    *,
    diffusion: float,
    branch_rate: float,
    mortality_rate: float,
    domain_sizes: tuple[float, ...],
    grid_sizes: tuple[int, ...],
    deadline: float,
    probe_position: float,
    relative_tolerance: float,
) -> pd.DataFrame:
    """Evaluate grid and far-boundary convergence at a fixed physical point."""
    rows: list[dict[str, float | int | bool]] = []
    values: list[float] = []
    cases = [(domain, nx) for domain in domain_sizes for nx in grid_sizes]
    for domain, nx in cases:
        solution = solve_backward_1d(
            diffusion=diffusion,
            branch_rate=branch_rate,
            mortality_rate=mortality_rate,
            domain_size=domain,
            deadline=deadline,
            nx=nx,
        )
        value = float(np.interp(probe_position, solution.x, solution.probability))
        values.append(value)
        rows.append(
            {
                "domain_size": domain,
                "nx": nx,
                "dx": solution.dx,
                "dt": solution.dt,
                "steps": solution.steps,
                "probe_position": probe_position,
                "probe_probability": value,
                "bound_residual": solution.bound_residual,
            }
        )
    reference = values[-1]
    for row, value in zip(rows, values, strict=True):
        error = abs(value - reference)
        relative = error / max(abs(reference), 1e-15)
        row["reference_probability"] = reference
        row["absolute_error"] = error
        row["relative_error"] = relative
        row["relative_tolerance"] = relative_tolerance
        row["passed"] = relative <= relative_tolerance
    return pd.DataFrame(rows)


def branching_brownian_monte_carlo(
    *,
    initial_position: float,
    diffusion: float,
    branch_rate: float,
    mortality_rate: float,
    domain_size: float,
    deadline: float,
    dt: float,
    replicates: int,
    rng: np.random.Generator,
    maximum_particles: int = 10_000,
) -> dict[str, float | int | bool]:
    """Simulate the same branching Brownian benchmark as the PDE.

    Branch and death events use exact per-step event probabilities. Motion uses
    Euler Gaussian increments, so comparison tolerance must include time-step
    error. Runs that hit the protective particle cap are reported as failures.
    """
    if not 0 < initial_position <= domain_size:
        raise ValueError("initial_position must lie inside the domain")
    if dt <= 0 or replicates <= 0:
        raise ValueError("dt and replicates must be positive")
    steps = math.ceil(deadline / dt)
    actual_dt = deadline / steps
    branch_probability = -math.expm1(-branch_rate * actual_dt)
    death_probability = -math.expm1(-mortality_rate * actual_dt)
    displacement_scale = math.sqrt(2.0 * diffusion * actual_dt)
    successes = 0
    cap_failures = 0
    for _ in range(replicates):
        positions = np.array([initial_position], dtype=float)
        hit = False
        for _ in range(steps):
            if positions.size == 0:
                break
            positions += displacement_scale * rng.normal(size=positions.size)
            if bool((positions <= 0.0).any()):
                hit = True
                break
            # Reflect repeatedly in case a large Gaussian increment crosses L.
            positions = domain_size - np.abs(
                (positions % (2.0 * domain_size)) - domain_size
            )
            # Undetected between-step Brownian-bridge crossings vanish as dt
            # decreases; dt convergence is mandatory for this benchmark.
            death = rng.random(positions.size) < death_probability
            positions = positions[~death]
            if positions.size:
                branches = rng.random(positions.size) < branch_probability
                positions = np.concatenate((positions, positions[branches]))
            if positions.size > maximum_particles:
                cap_failures += 1
                positions = np.empty(0)
                break
        successes += int(hit)
    low, high = wilson_interval(successes, replicates)
    return {
        "replicates": replicates,
        "successes": successes,
        "estimate": successes / replicates,
        "ci_low": low,
        "ci_high": high,
        "dt": actual_dt,
        "particle_cap_failures": cap_failures,
        "all_runs_valid": cap_failures == 0,
    }
