import numpy as np

from rootfpt.multiscale.soil import Grid2D
from rootfpt.multiscale.water import (
    ReducedWaterParameters,
    ReducedWaterState,
    run_m31_hydraulic_benchmark,
    step_reduced_water,
)


def test_reduced_water_balance_closes() -> None:
    grid = Grid2D(30, 24, (-3.0, 3.0), (0.0, 4.8))
    state = ReducedWaterState(grid, np.full((grid.nz, grid.nx), 0.3))
    parameters = ReducedWaterParameters(0.02, 0.08, 0.45, 0.03)
    root_density = np.zeros_like(state.water)
    root_density[5:18, 12:18] = 0.8
    for _ in range(30):
        step_reduced_water(
            state,
            parameters,
            root_length_density=root_density,
            dt=0.02,
            infiltration=0.01,
            evaporation=0.004,
        )
    assert state.maximum_balance_residual < 1e-12
    assert state.cumulative_root_gain > 0
    assert np.all(state.water >= parameters.residual_water)


def test_published_m31_hydraulic_benchmark_converges() -> None:
    coarse = run_m31_hydraulic_benchmark(50)
    fine = run_m31_hydraulic_benchmark(200)
    assert fine["relative_l2_error"] < coarse["relative_l2_error"]
    assert fine["relative_l2_error"] < 2e-5
    assert fine["kirchhoff_residual"] < 1e-9
