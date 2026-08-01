from __future__ import annotations

import math

import pytest

from rootfpt.theory import (
    BranchIncrement,
    classify_branch_sequence,
    folded_normal_laplace,
    marginal_probability,
    marginal_utility,
    poisson_success,
    small_angle_decorrelation,
)


def test_exact_marginal_identity() -> None:
    base, increment = 0.7, 0.25
    assert marginal_probability(base, increment) == pytest.approx(
        poisson_success(base + increment) - poisson_success(base)
    )
    assert marginal_utility(base, increment, 0.02) == pytest.approx(
        poisson_success(base + increment) - poisson_success(base) - 0.02
    )


@pytest.mark.parametrize(
    ("increments", "expected"),
    [
        ([BranchIncrement(0.01, 0.1)] * 3, "no_branching"),
        ([BranchIncrement(0.8, 0.01)] * 3, "boundary"),
        (
            [
                BranchIncrement(0.8, 0.02),
                BranchIncrement(0.3, 0.04),
                BranchIncrement(0.01, 0.2),
            ],
            "interior",
        ),
    ],
)
def test_regime_classification(
    increments: list[BranchIncrement],
    expected: str,
) -> None:
    regime, _, _ = classify_branch_sequence(0.2, increments)
    assert regime == expected


def test_folded_normal_degenerate_case() -> None:
    assert folded_normal_laplace(2.0, 0.0, 4.0) == pytest.approx(math.exp(-0.5))


def test_decorrelation_bounds_and_limits() -> None:
    low = small_angle_decorrelation(
        branch_spacing=0.1,
        persistence_length=10.0,
        correlation_length=1.0,
        search_radius=0.1,
    )
    high = small_angle_decorrelation(
        branch_spacing=3.0,
        persistence_length=1.0,
        correlation_length=0.2,
        search_radius=0.01,
    )
    for result in (low, high):
        assert 0 <= result["joint_lower_bound"] <= result["joint_upper_bound"] <= 1
    assert high["joint_upper_bound"] > low["joint_upper_bound"]
