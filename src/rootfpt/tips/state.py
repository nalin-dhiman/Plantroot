"""Active-tip and segment state for the reduced biological model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class TipStatus(StrEnum):
    ACTIVE = "active"
    DORMANT = "dormant"
    STOPPED = "stopped"
    DEAD = "dead"
    ABORTED = "aborted"


@dataclass
class TipState:
    tip_id: int
    parent_tip_id: int | None
    parent_segment_id: int | None
    root_order: int
    root_type: str
    position: np.ndarray
    orientation: float
    age: float
    radius: float
    arc_length: float
    status: TipStatus
    sensor_memory: np.ndarray
    emergence_time: float
    circumnutation_phase: float
    next_lateral_arc: float

    @property
    def direction(self) -> np.ndarray:
        return np.array([math.cos(self.orientation), math.sin(self.orientation)])


@dataclass(frozen=True)
class RootSegment:
    segment_id: int
    parent_segment_id: int | None
    producing_tip_id: int
    root_order: int
    start: tuple[float, float]
    end: tuple[float, float]
    radius: float
    created_time: float

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angular_difference(target: float, current: float) -> float:
    return wrap_angle(target - current)


def step_orientation(
    *,
    orientation: float,
    drift: float,
    rotational_diffusion: float,
    dt: float,
    rng: np.random.Generator,
) -> float:
    """Euler–Maruyama update for the 2-D angular SDE."""
    if rotational_diffusion < 0 or dt <= 0:
        raise ValueError("rotational_diffusion must be nonnegative and dt positive")
    noise = math.sqrt(2.0 * rotational_diffusion * dt) * float(rng.normal())
    return wrap_angle(orientation + drift * dt + noise)


def straight_step(position: np.ndarray, orientation: float, distance: float) -> np.ndarray:
    """Advance exactly along a unit orientation."""
    return np.asarray(position, dtype=float) + distance * np.array(
        [math.cos(orientation), math.sin(orientation)]
    )
