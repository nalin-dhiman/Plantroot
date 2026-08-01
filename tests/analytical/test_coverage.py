from __future__ import annotations

import numpy as np

from rootfpt.geometry.coverage import RectDomain, raster_metrics, verify_poisson_coverage

POINTS = np.array([[0.1, 0.1], [1.2, 0.2], [1.5, 0.9], [0.4, 0.8]])


def test_raster_tube_area_converges() -> None:
    domain = RectDomain(0.0, 2.0, 0.0, 2.0)
    coarse = raster_metrics(
        points=POINTS,
        radius=0.08,
        domain=domain,
        resolution=128,
        intensity_family="homogeneous",
        intensity_base=0.8,
    )
    fine = raster_metrics(
        points=POINTS,
        radius=0.08,
        domain=domain,
        resolution=256,
        intensity_family="homogeneous",
        intensity_base=0.8,
    )
    relative = abs(
        float(coarse["unique_tube_area"]) - float(fine["unique_tube_area"])
    ) / float(fine["unique_tube_area"])
    assert relative < 0.03
    assert float(fine["overlap_area"]) >= 0.0


def test_poisson_coverage_matches_void_law_smoke() -> None:
    table = verify_poisson_coverage(
        points=POINTS,
        radius=0.08,
        domain=RectDomain(0.0, 2.0, 0.0, 2.0),
        resolutions=(128, 256),
        families=("homogeneous",),
        intensity_base=0.8,
        replicates=5000,
        master_seed=912,
        probability_tolerance=0.04,
        area_relative_tolerance=0.03,
    )
    final = table.iloc[-1]
    assert bool(final["probability_passed"])
    assert bool(final["resolution_passed"])
