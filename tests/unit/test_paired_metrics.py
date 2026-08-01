from __future__ import annotations

import numpy as np

from rootfpt.metrics import paired_binary_summary, paired_rmst_summary


def test_paired_binary_summary_counts_discordance_and_is_deterministic() -> None:
    finite = np.array([1, 1, 0, 1, 0, 0])
    control = np.array([0, 1, 1, 0, 0, 0])
    first = paired_binary_summary(
        finite, control, bootstrap_seed=7, bootstrap_replicates=500
    )
    second = paired_binary_summary(
        finite, control, bootstrap_seed=7, bootstrap_replicates=500
    )
    assert first == second
    assert first["finite_only"] == 2
    assert first["control_only"] == 1
    assert first["paired_risk_difference"] == 1 / 6


def test_paired_rmst_uses_deadline_for_censoring() -> None:
    summary = paired_rmst_summary(
        np.array([2.0, np.nan]),
        np.array([1, 0]),
        np.array([3.0, 4.0]),
        np.array([1, 1]),
        deadline=10.0,
        bootstrap_seed=8,
        bootstrap_replicates=500,
    )
    assert summary["finite_rmst"] == 6.0
    assert summary["control_rmst"] == 3.5
