# ROOT-FPT

[![License: MIT](https://img.shields.io/badge/license-MIT-16865C.svg)](LICENSE)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-2563EB.svg)](pyproject.toml)
[![Version 1.1.1](https://img.shields.io/badge/version-1.1.1-7C3AED.svg)](CITATION.cff)
[![Testing app](https://img.shields.io/badge/testing-Open_Streamlit_app-FF4B4B.svg)](https://plantroot.streamlit.app/)

ROOT-FPT is research software for reproducible experiments with stochastic
root-tip motion, delayed branching, synthetic heterogeneous soils, explicit
root graphs, and terminal hydraulic analysis. It provides a Python package,
scripted workflows, and an interactive Streamlit application.

> **Scientific scope:** ROOT-FPT is a two-dimensional, synthetic, uncalibrated
> model. Presets are parameter configurations rather than plant species.
> Results are not field predictions, biological validation, or management
> recommendations.

![ROOT-FPT Explorer showing a synthetic root, controls, metrics, and depth profile](assets/app_preview.png)

## Capabilities

- Persistent stochastic tip growth with gravity, water, mechanical, and
  anisotropic soil responses.
- Developmentally delayed lateral emergence and explicit branching topology.
- Homogeneous, correlated, layered, deep-water, compacted, and cracked soil
  constructors.
- Geometry, depth-distribution, construction-accounting, and terminal
  hydraulic metrics.
- Deterministic named seeds for exact reruns and paired-environment
  comparisons.
- CSV, JSON, and ZIP exports with a SHA-256 signature for the ordered segment
  table.

![Six deterministic ROOT-FPT architecture presets in one controlled soil](assets/preset_gallery.png)

## Installation

ROOT-FPT supports Python 3.11 and 3.12.

```bash
git clone https://github.com/nalin-dhiman/Plantroot.git
cd Plantroot
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

## Using the explorer

The application provides four workspaces:

- **Explore** runs one reproducible root–soil realization and exports its graph
  and metrics.
- **Paired comparison** evaluates two architecture presets in identical
  synthetic soil realizations.
- **Diagnostics** compares 0.04-day and 0.02-day integrations and reports
  conservation residuals.
- **Model & limits** documents represented mechanisms and exclusions.

The standard development window is 1–5.5 days. A guarded extended preview
supports 7, 14, 21, and 30 days on a shared larger soil domain. Extended runs
are extrapolations of the short-window rules; they do not add root ageing,
turnover, seasonal forcing, or dynamic carbon limitation. The web interface
therefore limits long runs to 60 allocated tips and reports computational
truncation explicitly.

## Python example

```python
from rootfpt.explorer import result_signature, run_experiment

result = run_experiment(
    "dimorphic",
    "patchy_matern",
    seed=20260802,
    replicate=0,
    duration_days=5.5,
    dt_days=0.04,
)

print(result.metrics)
print(result_signature(result))
```

See the [API guide](API.md) and executable [tutorials](tutorials) for the
lower-level model components and paired workflows.

## Reproducible research

For every analysis, record the software version or Git commit, configuration,
master seed, replicate index, duration, integration step, and allocation cap.
Use paired environment seeds for architecture comparisons and ensembles for
uncertainty. A single visually selected realization is not representative.

The 5.5-day reference window has explicit resolution and conservation checks.
Longer previews should be treated as exploratory until their additional
biological mechanisms and numerical behavior are separately validated.

## Limitations

- Root development is represented in two spatial dimensions.
- Soil fields are controlled synthetic inputs, not measured profiles.
- The reduced water assay is not a Richards-equation solver.
- Terminal hydraulics does not feed water uptake back into development.
- Parameters are not calibrated to a species, genotype, site, or treatment.
- Root ageing, turnover, seasonal forcing, and dynamic carbon limitation are
  not represented.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .
python scripts/check_public_release.py
```

Changes to equations, units, stochastic coupling, or output schemas should
include focused tests and preserve deterministic results unless a numerical
correction is intentional and documented. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository structure

```text
app/                  Streamlit interface and deployment bootstrap
src/rootfpt/          simulation, analysis, hydraulics, and visualization
configs/              explicit synthetic experiment configurations
tests/                unit, analytical, stochastic, conservation, and UI tests
tutorials/            small executable examples
workflows/            scripted research workflows
scripts/              installation, verification, and software-asset tools
assets/               software-facing images used in this README
```

Additional documentation:

- [API guide](API.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Issue tracker](https://github.com/nalin-dhiman/Plantroot/issues)

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). For
reproducible use, cite the archived software release when available and record
the exact Git commit used for the analysis.

ROOT-FPT is released under the [MIT License](LICENSE).

© 2026 Nalin Dhiman, IIT Mandi
