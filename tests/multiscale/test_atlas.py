from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from workflows.run_root_soil_atlas import (
    CONFIG,
    build_grid,
    choose_medoids,
    seed_value,
    soil_from_config,
)


def atlas_config() -> dict:
    return yaml.safe_load(Path(CONFIG).read_text())


def test_environment_seed_is_paired_across_root_regimes() -> None:
    config = atlas_config()
    master = int(config["seed"])
    soil_index = list(config["soils"]).index("patchy_matern")
    seeds = {
        seed_value(master, 1, soil_index, 7)
        for _architecture in config["root_regimes"]
    }
    assert len(seeds) == 1


def test_all_declared_soils_share_grid_and_finite_fields() -> None:
    config = atlas_config()
    grid = build_grid(config)
    for index, soil_name in enumerate(config["soils"]):
        soil = soil_from_config(
            soil_name,
            grid,
            np.random.default_rng(index),
        )
        assert soil.grid == grid
        assert np.isfinite(soil.water).all()
        assert np.isfinite(soil.impedance).all()


def test_medoid_selection_uses_metric_space_not_seed_order() -> None:
    metrics = pd.DataFrame(
        {
            "architecture": ["a"] * 5,
            "soil": ["s"] * 5,
            "replicate": [0, 1, 2, 3, 4],
            "failed": [0] * 5,
            "maximum_depth": [0, 1, 2, 3, 20],
            "horizontal_spread": [0, 1, 2, 3, 20],
            "total_root_length": [0, 1, 2, 3, 20],
            "branch_count": [0, 1, 2, 3, 20],
            "mean_tortuosity": [0, 1, 2, 3, 20],
            "cumulative_collar_water": [0, 1, 2, 3, 20],
        }
    )
    rld = pd.DataFrame(
        {
            "architecture": np.repeat("a", 5),
            "soil": np.repeat("s", 5),
            "replicate": np.arange(5),
            "depth_bin": np.zeros(5, dtype=int),
            "root_length_density": [0, 1, 2, 3, 20],
        }
    )
    medoid = choose_medoids(metrics, rld)
    assert medoid.loc[0, "replicate"] == 2
