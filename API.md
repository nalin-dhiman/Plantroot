# ROOT-FPT API guide

The stable public surface for atlas-scale experiments is intentionally small.

## Persistent tips

`rootfpt.multiscale.agent.TipTraits` stores intrinsic elongation, rotational
diffusion and directional-response coefficients. `simulate_tip_ensemble(...)`
integrates independent persistent tips and returns positions, orientations,
speeds and first-passage times.

## Root architecture

`rootfpt.multiscale.architecture.RootType` combines tip traits with radius,
branch-spacing, delay, angle, mortality and successor-type settings.
`EmergenceResponse` defines the reduced potential-site response.
`simulate_architecture(...)` grows a branching `Architecture`, whose methods
provide geometry metrics and root-length density by depth.

## Soil

`rootfpt.multiscale.soil.Grid2D` defines a regular planar domain.
`SoilState` provides homogeneous and Matérn constructors plus layered, lens and
crack modifiers. Soil fields are controlled synthetic inputs, not a Richards
solver.

## Water and hydraulics

`hydraulic_architecture_solution(...)` solves the axial–radial network on a
fixed final architecture. `step_reduced_water(...)` advances the conservative
reduced water assay. Neither function constitutes full dynamic root–soil
uptake or rhizosphere-resolved flow.

## Atlas workflow helpers

`workflows/run_root_soil_atlas.py` exposes `soil_from_config(...)`,
`root_types_from_config(...)`, `simulate_one(...)`, `architecture_metrics(...)`
and `choose_medoids(...)`. They are workflow-level helpers rather than a
versioned Python package API, but the tutorials use them to guarantee exact
agreement with the frozen atlas design.

All stochastic calls require an explicit NumPy generator or a recorded seed.
See docstrings and `tests/multiscale/` for executable examples.

## Explorer API

`rootfpt.explorer.run_experiment(...)` is the stable high-level entry point used
by the web application. It returns an `ExperimentResult` containing the root
graph, soil state, named seeds, metrics, and depth-binned root-length density.
`segment_frame(...)` creates a tabular graph export, `result_archive(...)`
creates the browser download, and `result_signature(...)` hashes the ordered
segment table for exact repeatability checks.
