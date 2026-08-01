from __future__ import annotations

from pathlib import Path

from rootfpt.config import load_yaml, mutable_copy
from rootfpt.experiments.ensemble import DesignPoint, run_paired_design
from rootfpt.experiments.reduced import simulate_reduced


def _smoke_config() -> dict:
    root = Path(__file__).resolve().parents[2]
    config = mutable_copy(load_yaml(root / "configs" / "reduced" / "default.yaml"))
    config["time"]["t_max"] = 3.0
    config["numerics"]["maximum_steps"] = 100
    config["numerics"]["coverage_resolution"] = 32
    return config


def test_reduced_model_is_reproducible_and_budget_closed() -> None:
    config = _smoke_config()
    first = simulate_reduced(config, 123456)
    second = simulate_reduced(config, 123456)
    assert first.metrics == second.metrics
    assert len(first.segments) == len(second.segments)
    assert first.metrics["carbon_spent"] <= config["carbon"]["initial_budget"] + 1e-12
    assert first.metrics["carbon_closure_residual"] < config["numerics"]["tolerance"]
    assert first.metrics["oracle_sensor"] == 0


def test_paired_design_reuses_replicate_environment_seeds() -> None:
    config = _smoke_config()
    designs = (
        DesignPoint("low", (("root.hydrotropism", 0.0),)),
        DesignPoint("high", (("root.hydrotropism", 3.0),)),
    )
    frame = run_paired_design(
        base_config=config,
        designs=designs,
        replicates=2,
        master_seed=77,
        workers=1,
    )
    for _, group in frame.groupby("replicate"):
        assert group["replicate_master_seed"].nunique() == 1


def test_paired_design_accepts_locked_seed_manifest() -> None:
    config = _smoke_config()
    frame = run_paired_design(
        base_config=config,
        designs=(DesignPoint("control", (("branching.enabled", False),)),),
        replicates=2,
        master_seed=0,
        workers=1,
        replicate_seeds=[112233, 445566],
    )
    assert frame["replicate_master_seed"].tolist() == [112233, 445566]


def test_exact_maximum_step_boundary_is_not_a_numerical_failure() -> None:
    config = _smoke_config()
    config["time"]["dt"] = 0.04
    config["time"]["t_max"] = 0.08
    config["numerics"]["maximum_steps"] = 2
    result = simulate_reduced(config, 9876)
    assert result.metrics["failure_time"] == 0.08
    assert result.metrics["failed"] == 0


def test_explicit_no_branch_control_is_exact_and_reproducible() -> None:
    disabled = _smoke_config()
    disabled["branching"]["enabled"] = False
    disabled["branching"]["mean_spacing"] = 0.01
    first = simulate_reduced(disabled, 24680)
    second = simulate_reduced(disabled, 24680)

    assert first.metrics == second.metrics
    assert first.metrics["number_of_lateral_sites"] == 0
    assert first.metrics["number_of_branches"] == 0
    assert first.metrics["carbon_branching"] == 0.0
    assert first.metrics["carbon_closure_residual"] < disabled["numerics"]["tolerance"]


def test_no_branch_primary_axis_matches_large_spacing_approximation() -> None:
    disabled = _smoke_config()
    disabled["branching"]["enabled"] = False
    approximation = _smoke_config()
    approximation["branching"]["enabled"] = True
    approximation["branching"]["mean_spacing"] = 1.0e12

    exact = simulate_reduced(disabled, 13579)
    legacy = simulate_reduced(approximation, 13579)
    exact_primary = [
        (segment.start, segment.end)
        for segment in exact.segments
        if segment.producing_tip_id == 0
    ]
    legacy_primary = [
        (segment.start, segment.end)
        for segment in legacy.segments
        if segment.producing_tip_id == 0
    ]

    assert exact.metrics["number_of_lateral_sites"] == 0
    assert legacy.metrics["number_of_lateral_sites"] == 0
    assert exact_primary == legacy_primary


def test_hydraulic_passage_records_ordered_event_times() -> None:
    config = _smoke_config()
    config["hydraulics"] = {
        "enabled": True,
        "axial_conductivity": 1.0,
        "radial_conductance": 1.0,
        "collar_potential": 0.0,
        "soil_potential_scale": 1.0,
        "minimum_flow": 0.0,
        "minimum_volume": 0.0,
        "benefit_per_volume": 1.0e9,
    }
    config["success"]["geometric_threshold"] = 0.0
    config["success"]["resource_threshold"] = 0.0
    result = simulate_reduced(config, 86420)
    assert result.metrics["hydraulics_enabled"] == 1
    assert result.metrics["event_observed_hydraulic"] == 1
    assert result.metrics["event_observed_cumulative_use"] == 1
    assert result.metrics["T_geo"] <= result.metrics["T_hydraulic"]
    assert result.metrics["T_hydraulic"] <= result.metrics["T_cumulative_use"]
