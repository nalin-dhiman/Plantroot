"""Deterministic NumPy SeedSequence hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_STREAMS = (
    "environment",
    "development",
    "sensor",
    "observation",
    "optimization",
    "numerical_testing",
)


@dataclass
class RandomStreamManager:
    """Own independent named generators derived from one master seed."""

    master_seed: int
    names: tuple[str, ...] = DEFAULT_STREAMS
    _sequences: dict[str, np.random.SeedSequence] = field(init=False, repr=False)
    _generators: dict[str, np.random.Generator] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.master_seed < 0:
            raise ValueError("master_seed must be nonnegative")
        if len(self.names) != len(set(self.names)):
            raise ValueError("random stream names must be unique")
        root = np.random.SeedSequence(self.master_seed)
        children = root.spawn(len(self.names))
        self._sequences = dict(zip(self.names, children, strict=True))
        self._generators = {
            name: np.random.default_rng(sequence)
            for name, sequence in self._sequences.items()
        }

    def generator(self, name: str) -> np.random.Generator:
        """Return the generator for a declared stream."""
        try:
            return self._generators[name]
        except KeyError as exc:
            raise KeyError(f"unknown random stream {name!r}") from exc

    def manifest(self) -> dict[str, Any]:
        """Return exact entropy/spawn metadata without advancing a stream."""
        return {
            "master_seed": self.master_seed,
            "streams": {
                name: {
                    "entropy": sequence.entropy,
                    "spawn_key": list(sequence.spawn_key),
                    "pool_size": sequence.pool_size,
                    "state_words": sequence.generate_state(4).tolist(),
                }
                for name, sequence in self._sequences.items()
            },
        }

