"""Predeclared paired summaries for Phase 2 experiments."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def paired_binary_summary(
    finite: np.ndarray,
    control: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int = 20_000,
) -> dict[str, float | int]:
    """Summarize a paired binary contrast without dropping ties."""
    finite = np.asarray(finite, dtype=int)
    control = np.asarray(control, dtype=int)
    if finite.shape != control.shape or finite.ndim != 1 or finite.size == 0:
        raise ValueError("paired binary arrays must be nonempty and equally shaped")
    delta = finite - control
    finite_only = int(np.sum(delta == 1))
    control_only = int(np.sum(delta == -1))
    discordant = finite_only + control_only
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, finite.size, size=(bootstrap_replicates, finite.size))
    bootstrap = delta[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    p_value = (
        float(
            binomtest(
                min(finite_only, control_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "pairs": int(finite.size),
        "finite_success": float(finite.mean()),
        "control_success": float(control.mean()),
        "paired_risk_difference": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "finite_only": finite_only,
        "control_only": control_only,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": p_value,
        "probability_superiority": float(
            np.mean((delta > 0).astype(float) + 0.5 * (delta == 0))
        ),
    }


def paired_rmst_summary(
    finite_time: np.ndarray,
    finite_event: np.ndarray,
    control_time: np.ndarray,
    control_event: np.ndarray,
    *,
    deadline: float,
    bootstrap_seed: int,
    bootstrap_replicates: int = 20_000,
) -> dict[str, float]:
    """Compare restricted event times, assigning the deadline to censored runs."""
    finite = np.where(np.asarray(finite_event, dtype=bool), finite_time, deadline)
    control = np.where(np.asarray(control_event, dtype=bool), control_time, deadline)
    if finite.shape != control.shape or finite.ndim != 1:
        raise ValueError("paired RMST arrays must have equal one-dimensional shape")
    delta = finite - control
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, finite.size, size=(bootstrap_replicates, finite.size))
    low, high = np.quantile(delta[indices].mean(axis=1), (0.025, 0.975))
    return {
        "finite_rmst": float(finite.mean()),
        "control_rmst": float(control.mean()),
        "paired_rmst_difference": float(delta.mean()),
        "rmst_ci_low": float(low),
        "rmst_ci_high": float(high),
    }
