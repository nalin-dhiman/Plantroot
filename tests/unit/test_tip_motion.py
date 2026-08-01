from __future__ import annotations

import math

import numpy as np

from rootfpt.tips import step_orientation, straight_step


def test_straight_motion_is_exact_without_angular_change() -> None:
    start = np.array([0.2, 0.3])
    end = straight_step(start, math.pi / 2.0, 0.7)
    assert np.allclose(end, [0.2, 1.0], atol=1e-15)


def test_angular_diffusion_has_expected_variance() -> None:
    rng = np.random.default_rng(9)
    samples = np.array(
        [
            step_orientation(
                orientation=0.0,
                drift=0.0,
                rotational_diffusion=0.2,
                dt=0.1,
                rng=rng,
            )
            for _ in range(30_000)
        ]
    )
    assert abs(samples.mean()) < 0.005
    assert math.isclose(samples.var(), 2.0 * 0.2 * 0.1, rel_tol=0.04)
    directions = np.column_stack((np.cos(samples), np.sin(samples)))
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)

