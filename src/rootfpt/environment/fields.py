"""Transparent static and transient moisture environments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.ndimage import gaussian_filter


class MoistureEnvironment(Protocol):
    def value_gradient(self, position: np.ndarray, time: float) -> tuple[float, np.ndarray]: ...

    def patch_diagnostics(self, position: np.ndarray, time: float) -> dict[str, float]: ...


@dataclass(frozen=True)
class ResourcePatch:
    centre: tuple[float, float]
    amplitude: float
    sigma_x: float
    sigma_z: float
    birth_time: float
    lifetime: float
    decay_time: float

    def remaining_lifetime(self, time: float) -> float:
        return max(0.0, self.birth_time + self.lifetime - time)

    def active(self, time: float) -> bool:
        return self.birth_time <= time < self.birth_time + self.lifetime


@dataclass
class TransientPatchField:
    patches: tuple[ResourcePatch, ...]
    base_moisture: float = 0.05
    depth_gradient: float = 0.08

    def value_gradient(self, position: np.ndarray, time: float) -> tuple[float, np.ndarray]:
        x, z = np.asarray(position, dtype=float)
        value = self.base_moisture + self.depth_gradient * z
        gradient = np.array([0.0, self.depth_gradient])
        for patch in self.patches:
            if not patch.active(time):
                continue
            dx = x - patch.centre[0]
            dz = z - patch.centre[1]
            age = time - patch.birth_time
            temporal = math.exp(-age / patch.decay_time) if patch.decay_time > 0 else 1.0
            kernel = patch.amplitude * temporal * math.exp(
                -0.5 * ((dx / patch.sigma_x) ** 2 + (dz / patch.sigma_z) ** 2)
            )
            value += kernel
            gradient += kernel * np.array(
                [-dx / patch.sigma_x**2, -dz / patch.sigma_z**2]
            )
        return float(np.clip(value, 0.0, 1.0)), gradient

    def patch_diagnostics(self, position: np.ndarray, time: float) -> dict[str, float]:
        x, z = np.asarray(position, dtype=float)
        candidates: list[tuple[float, ResourcePatch]] = []
        for patch in self.patches:
            if patch.active(time):
                distance = math.hypot(
                    (x - patch.centre[0]) / patch.sigma_x,
                    (z - patch.centre[1]) / patch.sigma_z,
                )
                candidates.append((distance, patch))
        if not candidates:
            return {"resource": 0.0, "remaining_lifetime": 0.0, "distance": math.inf}
        distance, patch = min(candidates, key=lambda item: item[0])
        resource = patch.amplitude * math.exp(-0.5 * distance**2)
        return {
            "resource": resource,
            "remaining_lifetime": patch.remaining_lifetime(time),
            "distance": distance,
        }

    def oracle_direction(self, position: np.ndarray, time: float) -> np.ndarray:
        active = [patch for patch in self.patches if patch.active(time)]
        if not active:
            return np.zeros(2)
        target = max(
            active,
            key=lambda patch: patch.amplitude
            / (math.dist(tuple(position), patch.centre) + 1e-6),
        )
        vector = np.asarray(target.centre) - np.asarray(position)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else np.zeros(2)


@dataclass
class StaticCorrelatedField:
    values: np.ndarray
    x_limits: tuple[float, float]
    z_limits: tuple[float, float]

    @classmethod
    def generate(
        cls,
        *,
        rng: np.random.Generator,
        shape: tuple[int, int],
        correlation_cells: float,
        x_limits: tuple[float, float],
        z_limits: tuple[float, float],
    ) -> StaticCorrelatedField:
        raw = rng.normal(size=shape)
        values = gaussian_filter(raw, sigma=correlation_cells, mode="reflect")
        values -= values.min()
        values /= max(values.max(), 1e-12)
        return cls(values, x_limits, z_limits)

    def _coordinates(self, position: np.ndarray) -> tuple[float, float]:
        x, z = position
        fx = (x - self.x_limits[0]) / (self.x_limits[1] - self.x_limits[0])
        fz = (z - self.z_limits[0]) / (self.z_limits[1] - self.z_limits[0])
        return (
            np.clip(fx, 0.0, 1.0) * (self.values.shape[1] - 1),
            np.clip(fz, 0.0, 1.0) * (self.values.shape[0] - 1),
        )

    def value_gradient(self, position: np.ndarray, time: float) -> tuple[float, np.ndarray]:
        del time
        ix, iz = self._coordinates(position)
        x0, z0 = int(ix), int(iz)
        x1 = min(x0 + 1, self.values.shape[1] - 1)
        z1 = min(z0 + 1, self.values.shape[0] - 1)
        tx, tz = ix - x0, iz - z0
        value = (
            (1 - tx) * (1 - tz) * self.values[z0, x0]
            + tx * (1 - tz) * self.values[z0, x1]
            + (1 - tx) * tz * self.values[z1, x0]
            + tx * tz * self.values[z1, x1]
        )
        dz, dx = np.gradient(self.values)
        gradient = np.array([dx[z0, x0], dz[z0, x0]])
        return float(value), gradient

    def patch_diagnostics(self, position: np.ndarray, time: float) -> dict[str, float]:
        value, _ = self.value_gradient(position, time)
        return {"resource": value, "remaining_lifetime": math.inf, "distance": 0.0}


def scenario_patches(name: str) -> tuple[ResourcePatch, ...]:
    """Return deterministic scenario geometry; randomness is added by the runner."""
    scenarios = {
        "shallow_frequent": (
            ResourcePatch((-0.45, 0.35), 0.8, 0.25, 0.16, 0.0, 7.0, 5.0),
            ResourcePatch((0.45, 0.50), 0.7, 0.28, 0.18, 6.0, 7.0, 5.0),
            ResourcePatch((0.0, 0.65), 0.75, 0.30, 0.20, 12.0, 7.0, 5.0),
        ),
        "deep_infrequent": (
            ResourcePatch((0.0, 1.65), 0.9, 0.35, 0.22, 0.0, 24.0, 18.0),
        ),
        "mixed_depth": (
            ResourcePatch((-0.45, 0.45), 0.75, 0.26, 0.18, 0.0, 8.0, 5.0),
            ResourcePatch((0.35, 1.45), 0.85, 0.34, 0.24, 0.0, 24.0, 18.0),
        ),
        "patchy_static": (
            ResourcePatch((-0.48, 0.72), 0.75, 0.24, 0.20, 0.0, 1e9, 1e9),
            ResourcePatch((0.42, 1.28), 0.9, 0.28, 0.24, 0.0, 1e9, 1e9),
        ),
        "patchy_transient": (
            ResourcePatch((-0.48, 0.72), 0.78, 0.24, 0.20, 0.0, 12.0, 9.0),
            ResourcePatch((0.42, 1.28), 0.9, 0.28, 0.24, 4.0, 14.0, 10.0),
        ),
        "mechanically_obstructed": (
            ResourcePatch((0.55, 1.35), 0.9, 0.30, 0.24, 0.0, 20.0, 15.0),
        ),
    }
    try:
        return scenarios[name]
    except KeyError as exc:
        raise ValueError(f"unknown environment scenario {name!r}") from exc

