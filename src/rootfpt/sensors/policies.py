"""Biological local sensors and an explicitly unrealistic oracle bound."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from rootfpt.environment import MoistureEnvironment
from rootfpt.tips import TipState, angular_difference


class SensorPolicy(Protocol):
    name: str
    is_oracle: bool

    def turning_drift(
        self,
        *,
        tip: TipState,
        environment: MoistureEnvironment,
        time: float,
        dt: float,
        gain: float,
        rng: np.random.Generator,
    ) -> float: ...


def _gradient_turn(tip: TipState, gradient: np.ndarray, gain: float) -> float:
    norm = float(np.linalg.norm(gradient))
    if norm <= 1e-12:
        return 0.0
    target = math.atan2(float(gradient[1]), float(gradient[0]))
    saturation = norm / (0.25 + norm)
    return gain * saturation * math.sin(angular_difference(target, tip.orientation))


@dataclass
class NoSensor:
    name: str = "none"
    is_oracle: bool = False

    def turning_drift(
        self,
        *,
        tip: TipState,
        environment: MoistureEnvironment,
        time: float,
        dt: float,
        gain: float,
        rng: np.random.Generator,
    ) -> float:
        del tip, environment, time, dt, gain, rng
        return 0.0


@dataclass
class ReactiveSensor:
    noise: float
    name: str = "reactive"
    is_oracle: bool = False

    def turning_drift(
        self,
        *,
        tip: TipState,
        environment: MoistureEnvironment,
        time: float,
        dt: float,
        gain: float,
        rng: np.random.Generator,
    ) -> float:
        del dt
        _, gradient = environment.value_gradient(tip.position, time)
        observed = gradient + self.noise * rng.normal(size=2)
        return _gradient_turn(tip, observed, gain)


@dataclass
class MemorySensor:
    noise: float
    memory_time: float
    name: str = "memory"
    is_oracle: bool = False

    def turning_drift(
        self,
        *,
        tip: TipState,
        environment: MoistureEnvironment,
        time: float,
        dt: float,
        gain: float,
        rng: np.random.Generator,
    ) -> float:
        if self.memory_time <= 0:
            raise ValueError("memory_time must be positive")
        _, gradient = environment.value_gradient(tip.position, time)
        observed = gradient + self.noise * rng.normal(size=2)
        weight = 1.0 - math.exp(-dt / self.memory_time)
        tip.sensor_memory = (1.0 - weight) * tip.sensor_memory + weight * observed
        return _gradient_turn(tip, tip.sensor_memory, gain)


@dataclass
class DelayedSensor:
    noise: float
    delay: float
    name: str = "delayed"
    is_oracle: bool = False
    _history: dict[int, deque[tuple[float, np.ndarray]]] = field(
        default_factory=lambda: defaultdict(deque),
        repr=False,
    )

    def turning_drift(
        self,
        *,
        tip: TipState,
        environment: MoistureEnvironment,
        time: float,
        dt: float,
        gain: float,
        rng: np.random.Generator,
    ) -> float:
        del dt
        if self.delay < 0:
            raise ValueError("delay must be nonnegative")
        _, gradient = environment.value_gradient(tip.position, time)
        history = self._history[tip.tip_id]
        history.append((time, gradient + self.noise * rng.normal(size=2)))
        target_time = time - self.delay
        delayed = np.zeros(2)
        while len(history) > 1 and history[1][0] <= target_time:
            history.popleft()
        if history and history[0][0] <= target_time:
            delayed = history[0][1]
        return _gradient_turn(tip, delayed, gain)


@dataclass
class OracleSensor:
    """Unrealistic global-information upper bound."""

    name: str = "oracle"
    is_oracle: bool = True

    def turning_drift(
        self,
        *,
        tip: TipState,
        environment: MoistureEnvironment,
        time: float,
        dt: float,
        gain: float,
        rng: np.random.Generator,
    ) -> float:
        del dt, rng
        method = getattr(environment, "oracle_direction", None)
        if method is None:
            return 0.0
        direction = np.asarray(method(tip.position, time))
        return _gradient_turn(tip, direction, gain)


def build_sensor(
    policy: str,
    *,
    noise: float,
    memory_time: float,
    delay: float,
) -> SensorPolicy:
    if policy == "none":
        return NoSensor()
    if policy == "reactive":
        return ReactiveSensor(noise)
    if policy == "memory":
        return MemorySensor(noise, memory_time)
    if policy == "delayed":
        return DelayedSensor(noise, delay)
    if policy == "oracle":
        return OracleSensor()
    raise ValueError(f"unknown sensor policy {policy!r}")

