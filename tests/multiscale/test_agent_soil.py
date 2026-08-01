from dataclasses import replace

import numpy as np
import pytest

from rootfpt.multiscale.agent import (
    TipTraits,
    free_walk_msd,
    orientation_correlation,
    simulate_tip_ensemble,
)
from rootfpt.multiscale.soil import FIELD_UNITS, Grid2D, SoilState


def test_free_walk_analytical_limits() -> None:
    time = np.array([0.0, 0.5, 2.0])
    np.testing.assert_allclose(orientation_correlation(time, 0.7), np.exp(-0.7 * time))
    np.testing.assert_allclose(free_walk_msd(time, 2.0, 0.0), (2.0 * time) ** 2)
    assert free_walk_msd(100.0, 2.0, 0.7) / 100.0 == pytest.approx(2 * 4 / 0.7, rel=0.02)


def test_straight_agent_has_no_translational_brownian_noise() -> None:
    result = simulate_tip_ensemble(
        number=5,
        duration=2.0,
        dt=0.05,
        traits=TipTraits(speed=1.5, rotational_diffusion=0.0),
        rng=np.random.default_rng(1),
        initial_orientation=np.pi / 2,
    )
    np.testing.assert_allclose(result.final_positions[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.final_positions[:, 1], 3.0, atol=1e-10)


def test_matern_soil_has_documented_fields_and_cross_correlation() -> None:
    grid = Grid2D(48, 40, (-3.0, 3.0), (0.0, 5.0))
    soil = SoilState.matern(
        grid,
        rng=np.random.default_rng(12),
        correlation_length=0.7,
        cross_correlation=-0.65,
    )
    assert set(FIELD_UNITS) == {
        "water",
        "pressure_head",
        "hydraulic_conductivity",
        "impedance",
        "porosity",
        "anisotropy_xx",
        "anisotropy_xz",
        "anisotropy_zz",
        "nutrient",
        "oxygen",
    }
    assert abs(float(np.mean(soil.water)) - 0.28) < 0.01
    assert soil.empirical_cross_correlation() < -0.45
    assert np.all(soil.hydraulic_conductivity >= 0)


def test_bilinear_gradient_recovers_linear_field() -> None:
    grid = Grid2D(30, 32, (-2.0, 2.0), (0.0, 4.0))
    soil = SoilState.homogeneous(grid)
    horizontal, depth = grid.mesh
    soil = replace(soil, nutrient=2.0 * horizontal - 3.0 * depth)
    positions = np.array([[0.0, 2.0], [0.4, 1.5], [-0.7, 2.8]])
    gradient = soil.gradient("nutrient", positions)
    np.testing.assert_allclose(gradient, np.array([[2.0, -3.0]] * 3), atol=1e-10)
