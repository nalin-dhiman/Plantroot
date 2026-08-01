import numpy as np
import pytest

from rootfpt.multiscale.pair import (
    LineageTable,
    RelativePairGrid,
    TipSnapshot,
    binary_maximum_entropy_triplets,
    birth_death_factorial_moments,
    initialize_relative_pair,
    kirkwood_triplet_closure,
    lineage_mark,
    pair_records,
    radial_pair_correlation,
    relative_pair_mass,
    sister_relative_msd,
    step_relative_pair,
)


def test_lineage_marks_distinguish_sisters_and_shared_path() -> None:
    lineage = LineageTable(
        parent=np.array([-1, 0, 0]),
        birth_time=np.array([0.0, 1.0, 1.0]),
        path_at_birth=np.array([0.0, 2.0, 2.0]),
        branch_order=np.array([0, 1, 1]),
    )
    mark = lineage_mark(lineage, 1, 2)
    assert mark["relation"] == "sisters"
    assert mark["mrca_id"] == 0
    assert mark["graph_distance"] == 2
    assert mark["common_path_length"] == 0.0


def test_pair_records_are_exchange_unique() -> None:
    lineage = LineageTable(
        parent=np.array([-1, 0, 0]),
        birth_time=np.array([0.0, 1.0, 1.0]),
        path_at_birth=np.array([0.0, 2.0, 2.0]),
        branch_order=np.array([0, 1, 1]),
    )
    snapshot = TipSnapshot(
        positions=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        orientations=np.array([0.0, 0.0, np.pi / 2]),
        root_types=np.zeros(3, dtype=int),
        active_lineage_ids=np.arange(3),
        lineage=lineage,
        time=2.0,
    )
    records = pair_records(snapshot)
    assert len(records) == 3
    np.testing.assert_allclose(np.sort(records["distance"]), [1.0, 1.0, np.sqrt(2)])


def test_independent_periodic_points_have_unit_pair_correlation() -> None:
    rng = np.random.default_rng(82)
    estimates = []
    bins = np.linspace(0.1, 1.5, 9)
    for _ in range(80):
        positions = rng.uniform(0.0, 8.0, (250, 2))
        _, g2 = radial_pair_correlation(positions, bins, domain_size=(8.0, 8.0))
        estimates.append(g2)
    mean = np.mean(estimates, axis=0)
    np.testing.assert_allclose(mean, 1.0, atol=0.035)


def test_relative_pair_solver_preserves_symmetry_positivity_and_mass() -> None:
    grid = RelativePairGrid(256, (-8.0, 8.0))
    state = initialize_relative_pair(
        grid, pair_mass=3.0, mean_separation=1.0, separation_sigma=0.2
    )
    for _ in range(50):
        diagnostics = step_relative_pair(
            state, grid, dt=0.01, relative_diffusivity=0.4
        )
    assert relative_pair_mass(state, grid) == pytest.approx(3.0, rel=1e-12)
    assert float(np.min(state.density)) >= 0
    assert diagnostics["symmetry_error"] < 1e-12


def test_relative_pair_birth_and_mortality_accounting() -> None:
    grid = RelativePairGrid(128, (-6.0, 6.0))
    state = initialize_relative_pair(
        grid, pair_mass=2.0, mean_separation=0.0, separation_sigma=0.2
    )
    diagnostics = step_relative_pair(
        state,
        grid,
        dt=0.1,
        relative_diffusivity=0.2,
        mortality_rate=0.3,
        pair_birth_rate=0.5,
    )
    expected = 2.0 * np.exp(-0.06) + 0.5 * (1.0 - np.exp(-0.06)) / 0.6
    assert diagnostics["mass_after"] == pytest.approx(expected, rel=1e-12)


def test_sister_relative_msd_recovers_deterministic_angle_limit() -> None:
    time = np.linspace(0.0, 2.0, 10)
    angle = 0.4
    result = sister_relative_msd(
        time, speed=1.2, rotational_diffusion=0.0, half_branch_angle=angle
    )
    expected = (2.0 * 1.2 * time * np.sin(angle)) ** 2
    np.testing.assert_allclose(result, expected)


def test_birth_death_moments_include_ordered_distinct_pairs() -> None:
    time = np.array([0.0, 0.5, 1.0])
    first, second = birth_death_factorial_moments(
        time, birth_rate=0.7, mortality_rate=0.2
    )
    assert first[0] == 1.0
    assert second[0] == 0.0
    assert np.all(np.diff(first) > 0)
    assert np.all(np.diff(second) > 0)


def test_kirkwood_is_independence_for_factorized_pairs() -> None:
    first = np.array([0.2, 0.3, 0.5])
    second = np.outer(first, first)
    result = kirkwood_triplet_closure(first, second)
    np.testing.assert_allclose(result, np.einsum("i,j,k->ijk", first, first, first))


def test_binary_maximum_entropy_recovers_independent_triplets() -> None:
    first = np.array([0.2, 0.35, 0.5, 0.6])
    second = np.outer(first, first)
    third, residual = binary_maximum_entropy_triplets(first, second)
    expected = np.einsum("i,j,k->ijk", first, first, first)
    distinct = np.ones_like(third, dtype=bool)
    for i in range(len(first)):
        distinct[i, i, :] = False
        distinct[i, :, i] = False
        distinct[:, i, i] = False
    np.testing.assert_allclose(third[distinct], expected[distinct], atol=2e-6)
    assert residual < 2e-7
