"""Analytical tools for resource-constrained branching search."""

from rootfpt.theory.branching import (
    BranchIncrement,
    classify_branch_sequence,
    folded_normal_laplace,
    functional_exposure,
    marginal_probability,
    marginal_utility,
    poisson_success,
    small_angle_decorrelation,
)

__all__ = [
    "BranchIncrement",
    "classify_branch_sequence",
    "functional_exposure",
    "folded_normal_laplace",
    "marginal_probability",
    "marginal_utility",
    "poisson_success",
    "small_angle_decorrelation",
]
