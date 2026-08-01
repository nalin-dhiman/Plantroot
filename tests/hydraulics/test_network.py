from __future__ import annotations

import math

import numpy as np
import pytest

from rootfpt.hydraulics import (
    HydraulicSegment,
    SingularHydraulicSystem,
    maturation_multiplier,
    solve_hydraulic_network,
)


def test_one_uniform_segment() -> None:
    solution = solve_hydraulic_network(
        [HydraulicSegment(0, 1, length=2.0, axial_conductivity=4.0)],
        collar_node=0,
        collar_potential=1.0,
        fixed_potentials={1: 0.0},
    )
    assert math.isclose(solution.axial_flows[0], 2.0, rel_tol=1e-12)


def test_two_resistors_in_series() -> None:
    solution = solve_hydraulic_network(
        [
            HydraulicSegment(0, 1, 1.0, 2.0),
            HydraulicSegment(1, 2, 1.0, 1.0),
        ],
        collar_node=0,
        collar_potential=1.0,
        fixed_potentials={2: 0.0},
    )
    expected = 1.0 / (1.0 / 2.0 + 1.0)
    assert np.allclose(solution.axial_flows, expected, rtol=1e-12)


def test_symmetric_y_network() -> None:
    solution = solve_hydraulic_network(
        [HydraulicSegment(0, 1, 1.0, 2.0), HydraulicSegment(0, 2, 1.0, 2.0)],
        collar_node=0,
        collar_potential=0.0,
        fixed_potentials={1: 1.0, 2: 1.0},
    )
    assert np.allclose(solution.axial_flows, [-2.0, -2.0])


def test_asymmetric_y_network() -> None:
    solution = solve_hydraulic_network(
        [HydraulicSegment(0, 1, 1.0, 2.0), HydraulicSegment(0, 2, 1.0, 4.0)],
        collar_node=0,
        collar_potential=0.0,
        fixed_potentials={1: 1.0, 2: 1.0},
    )
    assert math.isclose(solution.axial_flows[1] / solution.axial_flows[0], 2.0)


def test_disconnected_terminal_fails_clearly() -> None:
    with pytest.raises(SingularHydraulicSystem):
        solve_hydraulic_network(
            [HydraulicSegment(0, 1, 1.0, 1.0)],
            collar_node=0,
            collar_potential=0.0,
            nodes=[0, 1, 2],
        )


def test_zero_radial_conductance_has_zero_flow_with_one_fixed_potential() -> None:
    solution = solve_hydraulic_network(
        [HydraulicSegment(0, 1, 1.0, 1.0, radial_conductance=0.0)],
        collar_node=0,
        collar_potential=-0.3,
    )
    assert np.allclose(solution.potentials, -0.3)
    assert np.allclose(solution.axial_flows, 0.0)


def test_large_axial_conductance_limit() -> None:
    solution = solve_hydraulic_network(
        [HydraulicSegment(0, 1, 1.0, 1e12, 1.0, 1.0)],
        collar_node=0,
        collar_potential=0.0,
    )
    assert abs(solution.potential(1) - solution.potential(0)) < 1e-12
    assert math.isclose(solution.collar_delivered_flow, 1.0, rel_tol=1e-10)


def test_age_dependent_conductance() -> None:
    young = maturation_multiplier(0.0, 2.0)
    old = maturation_multiplier(20.0, 2.0)
    assert young == 0.05
    assert old > 0.99
    solution = solve_hydraulic_network(
        [HydraulicSegment(0, 1, 1.0, old)],
        collar_node=0,
        collar_potential=1.0,
        fixed_potentials={1: 0.0},
    )
    assert math.isclose(solution.axial_flows[0], old)


def test_fixed_collar_potential_matches_one_segment_formula() -> None:
    gx, gr, soil = 2.0, 0.4, 1.0
    solution = solve_hydraulic_network(
        [HydraulicSegment(0, 1, 1.0, gx, gr, soil)],
        collar_node=0,
        collar_potential=0.0,
    )
    expected_terminal = 0.5 * gr * soil / (gx + 0.25 * gr)
    assert math.isclose(solution.potential(1), expected_terminal, rel_tol=1e-12)


def test_prescribed_collar_flux_is_recovered() -> None:
    solution = solve_hydraulic_network(
        [HydraulicSegment(0, 1, 1.0, 2.0, 1.0, 1.0)],
        collar_node=0,
        collar_flux=0.2,
    )
    assert math.isclose(solution.collar_delivered_flow, 0.2, rel_tol=1e-12)
    assert math.isclose(solution.radial_flows.sum(), 0.2, rel_tol=1e-12)


def test_segment_subdivision_preserves_axial_flow() -> None:
    whole = solve_hydraulic_network(
        [HydraulicSegment(0, 1, 2.0, 3.0)],
        collar_node=0,
        collar_potential=1.0,
        fixed_potentials={1: 0.0},
    )
    split = solve_hydraulic_network(
        [HydraulicSegment(0, 1, 1.0, 3.0), HydraulicSegment(1, 2, 1.0, 3.0)],
        collar_node=0,
        collar_potential=1.0,
        fixed_potentials={2: 0.0},
    )
    assert np.allclose(split.axial_flows, whole.axial_flows[0], rtol=1e-12)


def test_all_solutions_have_nonnegative_dissipation_and_small_residual() -> None:
    solution = solve_hydraulic_network(
        [
            HydraulicSegment(0, 1, 1.0, 2.0, 0.3, 1.0),
            HydraulicSegment(1, 2, 1.0, 1.0, 0.2, 0.5),
        ],
        collar_node=0,
        collar_potential=0.0,
    )
    assert solution.dissipation >= 0
    assert solution.relative_kirchhoff_residual < 1e-8
