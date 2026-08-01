from __future__ import annotations

import math

import numpy as np

from rootfpt.development import LateralDevelopment, LateralStatus
from rootfpt.tips import TipState, TipStatus


def _tip() -> TipState:
    return TipState(
        tip_id=0,
        parent_tip_id=None,
        parent_segment_id=None,
        root_order=0,
        root_type="primary",
        position=np.array([0.0, 1.0]),
        orientation=math.pi / 2.0,
        age=1.0,
        radius=0.03,
        arc_length=1.0,
        status=TipStatus.ACTIVE,
        sensor_memory=np.zeros(2),
        emergence_time=0.0,
        circumnutation_phase=0.0,
        next_lateral_arc=0.5,
    )


def test_lateral_site_emerges_behind_continuing_parent() -> None:
    development = LateralDevelopment(
        mean_spacing=10.0,
        spacing_shape=2.0,
        mean_emergence_delay=0.0,
        abortion_probability=0.0,
        dormancy_probability=0.0,
        daughter_angle_mean=0.7,
        daughter_angle_sd=0.0,
        maximum_order=2,
        rng=np.random.default_rng(3),
    )
    tip = _tip()
    sites = development.register_growth(
        tip=tip,
        segment_id=10,
        start_position=np.array([0.0, 0.0]),
        end_position=np.array([0.0, 1.0]),
        previous_arc_length=0.0,
        time=1.0,
    )
    assert len(sites) == 1
    assert sites[0].status == LateralStatus.SCHEDULED
    assert np.allclose(sites[0].position, [0.0, 0.5])
    assert development.due_sites(1.0) == sites
    assert tip.status == TipStatus.ACTIVE
    assert np.allclose(tip.position, [0.0, 1.0])

