import numpy as np
import pytest

from rootfpt.multiscale.p1 import (
    P1Parameters,
    initialize_p1_state,
    p1_characteristic_speed,
    p1_mass,
    p1_msd,
    step_p1,
)
from rootfpt.multiscale.soil import Grid2D


def parameters(number_types: int = 1, *, speed: float = 1.0, dr: float = 1.0) -> P1Parameters:
    return P1Parameters(
        speeds=np.full(number_types, speed),
        rotational_diffusion=np.full(number_types, dr),
        mortality=np.zeros(number_types),
        branching_rate=np.zeros(number_types),
        transition=np.zeros((number_types, number_types)),
        daughter_polarization=np.zeros(number_types),
        force=np.zeros((number_types, 2)),
    )


def test_p1_coefficients_recover_ballistic_and_diffusive_limits() -> None:
    speed = 1.7
    dr = 0.8
    short_time = 1e-4
    long_time = 100.0 / dr
    assert p1_msd(short_time, speed, dr) == pytest.approx(
        (speed * short_time) ** 2, rel=1e-4
    )
    effective_diffusivity = speed**2 / (2.0 * dr)
    assert p1_msd(long_time, speed, dr) == pytest.approx(
        4.0 * effective_diffusivity * long_time, rel=0.011
    )
    assert p1_characteristic_speed(speed) == pytest.approx(speed / np.sqrt(2.0))


def test_p1_periodic_solver_is_positive_and_conservative() -> None:
    grid = Grid2D(48, 48, (-4.0, 4.0), (-4.0, 4.0))
    state = initialize_p1_state(
        grid,
        count_by_type=np.array([3.0]),
        centre=(0.0, 0.0),
        spatial_sigma=0.45,
        mean_angle=0.3,
        angle_concentration=0.6,
    )
    initial = p1_mass(state, grid)
    for _ in range(40):
        step_p1(state, grid, parameters(), dt=0.01)
    np.testing.assert_allclose(p1_mass(state, grid), initial, rtol=2e-12, atol=2e-12)
    assert float(np.min(state.density)) >= 0.0
    assert np.sum(state.root_length_density) * grid.cell_area == pytest.approx(1.2, rel=1e-12)


def test_p1_branching_injects_declared_daughter_mass() -> None:
    grid = Grid2D(32, 32, (-3.0, 3.0), (-3.0, 3.0))
    state = initialize_p1_state(
        grid,
        count_by_type=np.array([2.0, 0.0]),
        centre=(0.0, 0.0),
        spatial_sigma=0.4,
    )
    base = parameters(2, speed=0.0, dr=0.2)
    values = P1Parameters(
        speeds=base.speeds,
        rotational_diffusion=base.rotational_diffusion,
        mortality=base.mortality,
        branching_rate=np.array([0.5, 0.0]),
        transition=np.array([[0.0, 1.0], [0.0, 0.0]]),
        daughter_polarization=np.array([0.7, 0.0]),
        force=base.force,
    )
    step_p1(state, grid, values, dt=0.02)
    np.testing.assert_allclose(p1_mass(state, grid), [2.0, 0.02], rtol=1e-12, atol=1e-12)


def test_p1_reflecting_boundary_has_no_mass_loss() -> None:
    grid = Grid2D(36, 36, (-2.0, 2.0), (0.0, 4.0))
    state = initialize_p1_state(
        grid,
        count_by_type=np.array([1.0]),
        centre=(0.0, 0.5),
        spatial_sigma=0.25,
        mean_angle=np.pi / 2,
        angle_concentration=0.4,
    )
    initial = p1_mass(state, grid)
    for _ in range(40):
        step_p1(
            state,
            grid,
            parameters(dr=0.5),
            dt=0.01,
            x_boundary="reflecting",
            z_boundary="reflecting",
        )
    np.testing.assert_allclose(p1_mass(state, grid), initial, rtol=2e-12, atol=2e-12)
