from __future__ import annotations

import math

import numpy as np
import pytest

from rootfpt.clustered import (
    EqualBudget,
    Grid2D,
    action_networks,
    capacity_truncated_mean,
    cluster_metrics,
    cluster_probability_map,
    kernel_on_grid,
    rasterize_network,
)


def test_equal_budget_actions_spend_exactly_the_same_material() -> None:
    budget = EqualBudget(
        step_budget=1.3,
        length_cost=0.9,
        maintenance_cost=0.2,
        residence_time=0.7,
        branch_initiation_cost=0.17,
    )
    assert budget.spent("extend") == pytest.approx(1.3)
    assert budget.spent("branch") == pytest.approx(1.3)
    assert budget.branch_total_length < budget.extension_length


@pytest.mark.parametrize("family", ["thomas", "matern"])
def test_offspring_kernel_is_normalized(family: str) -> None:
    grid = Grid2D(resolution=96)
    kernel = kernel_on_grid(grid, 0.35, family)
    assert float(kernel.sum() * grid.cell_area) == pytest.approx(1.0)


@pytest.mark.parametrize("family", ["deterministic", "exponential", "gamma"])
def test_capacity_truncation_limits(family: str) -> None:
    values = capacity_truncated_mean(
        np.asarray([0.0, 1e-8, 1e3]),
        family=family,
        mean_capacity=1.7,
    )
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1e-8, rel=2e-5)
    assert values[2] == pytest.approx(1.7, rel=1e-6)


def test_poisson_limit_recovers_tube_area() -> None:
    grid = Grid2D(lower=-3.0, upper=3.0, resolution=192)
    geometry = action_networks(EqualBudget(), branch_angle=math.pi / 2)["extend"]
    mask = rasterize_network(geometry, grid, 0.12)
    p = cluster_probability_map(mask, grid=grid, sigma=0.3, family="thomas")
    lam = 0.42
    mu = 0.002
    metrics = cluster_metrics(
        p,
        grid=grid,
        parent_intensity=lam / mu,
        mean_offspring=mu,
    )
    area = float(mask.sum() * grid.cell_area)
    assert metrics["expected_distinct_clusters"] == pytest.approx(lam * area, rel=2e-3)


def test_cluster_saturation_is_below_raw_micro_site_count() -> None:
    grid = Grid2D(resolution=128)
    geometry = action_networks(EqualBudget(), branch_angle=math.pi / 2)["branch"]
    mask = rasterize_network(geometry, grid, 0.15)
    p = cluster_probability_map(mask, grid=grid, sigma=0.25, family="matern")
    metrics = cluster_metrics(
        p,
        grid=grid,
        parent_intensity=0.7,
        mean_offspring=8.0,
        search_area=float(mask.sum() * grid.cell_area),
    )
    assert metrics["expected_distinct_clusters"] < metrics["expected_micro_site_contacts"]
