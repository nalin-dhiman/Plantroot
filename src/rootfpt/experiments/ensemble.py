"""Paired experiment designs over the reduced simulator."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rootfpt.experiments.reduced import simulate_reduced


@dataclass(frozen=True)
class DesignPoint:
    name: str
    overrides: tuple[tuple[str, float | int | str], ...]


def apply_overrides(config: dict[str, Any], overrides: tuple[tuple[str, Any], ...]) -> dict:
    result = copy.deepcopy(config)
    for path, value in overrides:
        keys = path.split(".")
        target = result
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return result


def paired_replicate_seeds(master_seed: int, count: int) -> list[int]:
    sequences = np.random.SeedSequence(master_seed).spawn(count)
    return [int(sequence.generate_state(1, dtype=np.uint64)[0]) for sequence in sequences]


def _run_design_job(
    job: tuple[str, tuple[tuple[str, Any], ...], dict[str, Any], int, int]
) -> dict[str, Any]:
    name, overrides, base_config, replicate, seed = job
    config = apply_overrides(base_config, overrides)
    result = simulate_reduced(config, seed)
    return {
        "design": name,
        "replicate": replicate,
        **{path.replace(".", "__"): value for path, value in overrides},
        **result.metrics,
    }


def run_paired_design(
    *,
    base_config: dict[str, Any],
    designs: tuple[DesignPoint, ...],
    replicates: int,
    master_seed: int,
    workers: int,
    replicate_seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Run each design on the same set of replicate master seeds."""
    seeds = (
        paired_replicate_seeds(master_seed, replicates)
        if replicate_seeds is None
        else [int(seed) for seed in replicate_seeds]
    )
    if len(seeds) != replicates:
        raise ValueError("replicate_seeds length must equal replicates")
    if len(seeds) != len(set(seeds)):
        raise ValueError("replicate_seeds must be unique")
    jobs = [
        (design.name, design.overrides, base_config, replicate, seed)
        for design in designs
        for replicate, seed in enumerate(seeds)
    ]
    if workers == 1:
        rows = [_run_design_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_design_job, jobs))
    return pd.DataFrame(rows)
