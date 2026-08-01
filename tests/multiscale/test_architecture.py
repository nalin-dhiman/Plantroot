import numpy as np

from rootfpt.multiscale.agent import TipTraits
from rootfpt.multiscale.architecture import (
    EmergenceResponse,
    RootType,
    SiteStatus,
    _CoupledBrownianPath,
    _dry_memory_decay,
    _KeyedRandom,
    simulate_architecture,
)
from rootfpt.multiscale.soil import Grid2D, SoilState


def test_lateral_emergence_does_not_fission_parent_tip() -> None:
    grid = Grid2D(60, 60, (-4.0, 4.0), (0.0, 8.0))
    soil = SoilState.homogeneous(grid, impedance=0.05, water=0.4, nutrient=1.0)
    primary = RootType(
        "primary",
        TipTraits(speed=2.0, rotational_diffusion=1e-8),
        0.08,
        0.25,
        8.0,
        0.0,
        0.7,
        0.01,
        successor="lateral",
    )
    lateral = RootType(
        "lateral",
        TipTraits(speed=1.0, rotational_diffusion=1e-8),
        0.04,
        10.0,
        8.0,
        0.0,
        0.7,
        0.01,
        successor=None,
    )
    architecture = simulate_architecture(
        soil=soil,
        root_types={"primary": primary, "lateral": lateral},
        primary_type="primary",
        response=EmergenceResponse(
            intercept=20.0,
            abortion_probability=0.0,
            dormancy_probability=0.0,
        ),
        duration=2.0,
        dt=0.05,
        rng=np.random.default_rng(2),
        max_order=1,
    )
    emerged = [site for site in architecture.sites if site.status == SiteStatus.EMERGED]
    assert emerged
    first_emergence = min(site.emergence_time for site in emerged)
    assert any(
        segment.order == 0 and segment.created_time > first_emergence
        for segment in architecture.segments
    )
    assert any(segment.order == 1 for segment in architecture.segments)
    assert architecture.metrics()["branch_count"] > 0


def test_dry_memory_decay_has_a_physical_timescale() -> None:
    assert np.isclose(_dry_memory_decay(0.08), 0.95)
    assert np.isclose(_dry_memory_decay(0.04) ** 2, _dry_memory_decay(0.08))
    assert np.isclose(_dry_memory_decay(0.02) ** 4, _dry_memory_decay(0.08))


def test_coupled_brownian_path_is_additive_across_nested_steps() -> None:
    path = _CoupledBrownianPath(_KeyedRandom(481), duration=1.0, quantum=0.02)
    coarse = path.increment(3, 0.0, 0.08)
    fine = sum(path.increment(3, start, start + 0.02) for start in np.arange(0.0, 0.08, 0.02))
    assert np.isclose(coarse, fine, atol=1e-14)

    off_grid = path.increment(7, 0.013, 0.08)
    partitioned = path.increment(7, 0.013, 0.04) + path.increment(7, 0.04, 0.08)
    assert np.isclose(off_grid, partitioned, atol=1e-14)
    assert np.isfinite(path.value(7, 1.0))


def test_branch_sites_are_interpolated_and_emerge_within_step() -> None:
    grid = Grid2D(80, 80, (-4.0, 4.0), (0.0, 8.0))
    soil = SoilState.homogeneous(grid, impedance=0.05, water=0.4, nutrient=1.0)
    primary = RootType(
        "primary",
        TipTraits(speed=2.0, rotational_diffusion=1e-8),
        0.08,
        0.23,
        1e8,
        0.0,
        0.7,
        0.0,
        successor="lateral",
    )
    lateral = RootType(
        "lateral",
        TipTraits(speed=1.0, rotational_diffusion=1e-8),
        0.04,
        10.0,
        1e8,
        0.0,
        0.7,
        0.0,
    )
    architecture = simulate_architecture(
        soil=soil,
        root_types={"primary": primary, "lateral": lateral},
        primary_type="primary",
        response=EmergenceResponse(
            intercept=20.0,
            abortion_probability=0.0,
            dormancy_probability=0.0,
        ),
        duration=1.2,
        dt=0.2,
        rng=np.random.default_rng(19),
        max_order=1,
    )
    assert architecture.sites
    assert all(
        np.allclose(site.position, architecture.nodes[site.parent_node])
        for site in architecture.sites
    )
    assert any(
        not np.isclose(site.initiation_time / 0.2, round(site.initiation_time / 0.2))
        for site in architecture.sites
    )
    assert any(
        segment.order == 1
        and segment.created_time <= min(site.initiation_time for site in architecture.sites) + 0.2
        for segment in architecture.segments
    )


def test_unbranched_length_is_invariant_to_final_partial_step() -> None:
    grid = Grid2D(100, 100, (-10.0, 10.0), (0.0, 20.0))
    soil = SoilState.homogeneous(grid, impedance=0.05, water=0.4, nutrient=1.0)
    root = RootType(
        "primary",
        TipTraits(speed=1.0, rotational_diffusion=1e-8),
        0.08,
        100.0,
        1e8,
        0.0,
        0.0,
        0.0,
    )
    lengths = []
    depths = []
    for dt in (0.08, 0.04, 0.02):
        architecture = simulate_architecture(
            soil=soil,
            root_types={"primary": root},
            primary_type="primary",
            response=EmergenceResponse(),
            duration=1.1,
            dt=dt,
            rng=np.random.default_rng(918),
            max_order=0,
        )
        metrics = architecture.metrics()
        lengths.append(metrics["total_length"])
        depths.append(metrics["depth"])
    assert np.ptp(lengths) < 1e-10
    assert np.ptp(depths) / np.mean(depths) < 1e-4
