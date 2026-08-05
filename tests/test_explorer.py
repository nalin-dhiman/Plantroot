from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from rootfpt.explorer import (
    MAX_EXPLORER_DURATION_DAYS,
    relative_change,
    result_archive,
    result_signature,
    run_experiment,
    segment_frame,
)
from rootfpt.visualization.explorer import architecture_figure


def short_run(architecture: str = "dimorphic"):
    return run_experiment(
        architecture,
        "patchy_matern",
        seed=93,
        replicate=2,
        duration_days=1.0,
        dt_days=0.08,
        max_tips=40,
    )


def test_explorer_run_is_exactly_reproducible() -> None:
    first = short_run()
    second = short_run()
    assert result_signature(first) == result_signature(second)
    assert first.metrics == second.metrics


def test_explorer_pair_reuses_environment_seed() -> None:
    first = short_run("taproot")
    second = short_run("fibrous")
    assert first.seeds["environment_seed"] == second.seeds["environment_seed"]
    assert first.seeds["growth_seed"] != second.seeds["growth_seed"]


def test_export_contains_analysis_ready_files() -> None:
    result = short_run()
    with zipfile.ZipFile(io.BytesIO(result_archive(result))) as archive:
        assert set(archive.namelist()) == {
            "README.txt",
            "metrics.json",
            "root_length_density.csv",
            "segments.csv",
        }
        assert len(archive.read("segments.csv")) > 40
    assert not segment_frame(result).empty


def test_architecture_plot_has_separate_colorbar_and_no_embedded_figure_number() -> None:
    figure = architecture_figure(short_run(), "Dimorphic", "Patchy water")
    assert len(figure.axes) == 2
    assert "Figure" not in figure.axes[0].get_title()


def test_relative_change_is_symmetric() -> None:
    assert relative_change(4.0, 5.0) == relative_change(5.0, 4.0) == 20.0


def test_extended_horizon_expands_domain_without_changing_grid_spacing() -> None:
    result = run_experiment(
        "taproot",
        "patchy_matern",
        seed=17,
        duration_days=7.0,
        dt_days=0.08,
        max_tips=40,
    )
    scale = float(result.settings["domain_scale"])
    assert scale > 1.0
    assert result.soil.grid.z_limits == pytest.approx((0.0, 12.0 * scale))
    assert result.soil.grid.dz == pytest.approx(12.0 / 48.0, rel=0.02)
    assert result.metrics["maximum_depth_cm"] <= result.depth_bins[-1]
    assert result.depth_bins[-1] <= result.soil.grid.z_limits[1]

    later = run_experiment(
        "taproot",
        "patchy_matern",
        seed=17,
        duration_days=14.0,
        dt_days=0.08,
        max_tips=40,
    )
    assert later.soil.grid == result.soil.grid
    np.testing.assert_array_equal(later.soil.water, result.soil.water)
    np.testing.assert_array_equal(later.soil.impedance, result.soil.impedance)


def test_explorer_rejects_horizons_beyond_interactive_limit() -> None:
    with pytest.raises(ValueError, match="duration_days"):
        run_experiment(
            "taproot",
            "homogeneous",
            duration_days=MAX_EXPLORER_DURATION_DAYS + 0.1,
        )
