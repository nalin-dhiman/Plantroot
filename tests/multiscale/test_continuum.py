import numpy as np
import pytest

from rootfpt.multiscale.continuum import (
    KineticGrid,
    KineticParameters,
    derived_diffusion_coefficients,
    initialize_kinetic_state,
    kinetic_mass,
    step_kinetic,
)
from rootfpt.multiscale.soil import Grid2D


def test_derived_diffusion_coefficients_are_not_fitted() -> None:
    result = derived_diffusion_coefficients(
        np.array([2.0, 1.0]),
        np.array([0.5, 2.0]),
        np.array([[0.2, -0.1], [0.0, 0.4]]),
    )
    np.testing.assert_allclose(result.diffusivity, [4.0, 0.25])
    np.testing.assert_allclose(result.drift, [[0.4, -0.2], [0.0, 0.1]])


def test_periodic_kinetic_solver_preserves_tip_mass_and_positivity() -> None:
    spatial = Grid2D(24, 24, (-3.0, 3.0), (-3.0, 3.0))
    grid = KineticGrid(spatial, ntheta=16, nage=40, age_step=0.01)
    state = initialize_kinetic_state(
        grid,
        number_types=1,
        count_by_type=np.array([3.0]),
        centre=(0.0, 0.0),
        spatial_sigma=0.25,
        mean_angle=0.4,
        angle_concentration=3.0,
    )
    parameters = KineticParameters(
        speeds=np.array([1.0]),
        rotational_diffusion=np.array([0.8]),
        mortality=np.array([0.0]),
        branching_rate=np.array([0.0]),
        transition=np.zeros((1, 1)),
        branch_angle=np.array([0.0]),
        branch_concentration=np.array([0.0]),
    )
    initial = kinetic_mass(state, grid)
    for _ in range(20):
        step_kinetic(state, grid, parameters, dt=0.01)
    np.testing.assert_allclose(kinetic_mass(state, grid), initial, rtol=2e-10, atol=1e-11)
    assert float(np.min(state.density)) >= 0.0
    expected_length = 3.0 * 1.0 * 0.2
    assert np.sum(state.root_length_density) * spatial.cell_area == pytest.approx(
        expected_length, rel=1e-8
    )


def test_branching_boundary_adds_daughters_without_removing_parent() -> None:
    spatial = Grid2D(16, 16, (-2.0, 2.0), (-2.0, 2.0))
    grid = KineticGrid(spatial, ntheta=12, nage=20, age_step=0.02)
    state = initialize_kinetic_state(
        grid,
        number_types=2,
        count_by_type=np.array([2.0, 0.0]),
        centre=(0.0, 0.0),
        spatial_sigma=0.2,
    )
    parameters = KineticParameters(
        speeds=np.array([0.0, 0.0]),
        rotational_diffusion=np.array([0.2, 0.2]),
        mortality=np.zeros(2),
        branching_rate=np.array([1.0, 0.0]),
        transition=np.array([[0.0, 1.0], [0.0, 0.0]]),
        branch_angle=np.array([0.7, 0.0]),
        branch_concentration=np.array([20.0, 0.0]),
    )
    step_kinetic(state, grid, parameters, dt=0.02)
    mass = kinetic_mass(state, grid)
    assert mass[0] == pytest.approx(2.0, rel=1e-9)
    assert mass[1] == pytest.approx(0.04, rel=1e-6)
