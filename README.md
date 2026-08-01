# ROOT-FPT Explorer

[![license: MIT](https://img.shields.io/badge/license-MIT-16865C.svg)](LICENSE)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-2563EB.svg)](pyproject.toml)

ROOT-FPT is open research software for controlled experiments with persistent
stochastic root-tip motion, delayed branching, synthetic heterogeneous soils,
explicit root graphs, and terminal hydraulic analysis. The Streamlit explorer
runs single realizations, small paired comparisons, numerical smoke tests, and
analysis-ready exports.

> **Scientific boundary:** this is a two-dimensional, synthetic, uncalibrated
> model. Preset names are parameter combinations, not species. Outputs are not
> field predictions, biological validation, or management advice.

![ROOT-FPT Explorer showing a synthetic root, controls, metrics, and depth profile](assets/app_preview.png)

## Install and run

Python 3.11 or 3.12 is required. Use `python -m ...` commands so installation
and Streamlit always use the same interpreter.

```bash
git clone https://github.com/nalin-dhiman/Plantroot.git
cd Plantroot
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[app]"
python scripts/doctor.py
python -m streamlit run streamlit_app.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

The installation is complete only after pip reports a successful install.
Dependency resolution may take a few minutes on a new machine.

### If the repository already exists

Do not clone over an existing directory. Update the existing checkout:

```bash
cd Plantroot
git pull --ff-only
source .venv/bin/activate
python -m pip install -e ".[app]"
python scripts/doctor.py
python -m streamlit run streamlit_app.py
```

If `.venv` does not exist, create it using the clean-install commands above.

### Fixing `ModuleNotFoundError: rootfpt`

This usually means pip was interrupted or `streamlit` came from a different
Python installation. Check the active tools:

```bash
which python
python -m pip show rootfpt
python -c "import rootfpt; print(rootfpt.__file__)"
python -m streamlit version
```

Then rerun the installation without interrupting it:

```bash
python -m pip install -e ".[app]"
python scripts/doctor.py
python -m streamlit run streamlit_app.py
```

The app also resolves `src/rootfpt` directly when launched from a source
checkout, but a complete installation is still recommended for reproducible
work.

## What can be explored?

![Six deterministic ROOT-FPT architecture presets grown in one controlled soil](assets/preset_gallery.png)

The presets above use the same simulator, domain, duration, and homogeneous
soil constructor. They are deliberately contrasting configurations—not claims
about real taxa or optimal root systems.

The app has four workspaces:

- **Explore:** grow one deterministic-by-seed architecture, inspect its graph
  and depth profile, and download metrics and segments.
- **Paired comparison:** compare two presets against identical synthetic soil
  realizations. Interactive samples are intentionally small and exploratory.
- **Diagnostics:** compare 0.04-day and 0.02-day integrations and inspect
  hydraulic and construction-accounting residuals.
- **Model & limits:** see which mechanisms are represented and which are not.

Downloads contain settings, named seeds, metrics, the segment table, the
root-length-density profile, and an exact segment-table SHA-256.

## Python API

```python
from rootfpt.explorer import result_signature, run_experiment, segment_frame

run = run_experiment(
    "dimorphic",
    "patchy_matern",
    seed=20260802,
    replicate=0,
    duration_days=5.5,
    dt_days=0.04,
)

print(run.metrics)
print(result_signature(run))
segments = segment_frame(run)
```

See [API.md](API.md) and [`tutorials/`](tutorials) for additional examples.

## Research-use checklist

- Record the repository commit, Python version, configuration, seed, and
  replicate index.
- Use paired environment seeds when comparing architecture presets.
- Use ensembles for uncertainty; a visually interesting realization is not a
  representative sample.
- Check numerical resolution for conclusions sensitive to event timing.
- Retain failures and define exclusions before running an experiment.
- Treat hydraulic and construction outputs as model indices unless separately
  calibrated and validated.

## Known limitations

- Root development is represented in two spatial dimensions.
- Soil constructors are controlled inputs rather than empirical profiles.
- The reduced soil-water assay is not Richards flow.
- Terminal hydraulic analysis does not feed water uptake back into development.
- Parameters are not calibrated to a species, genotype, field site, or treatment.
- Large ensembles belong in scripted workflows, not the shared web interface.

## Development and verification

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .
python scripts/check_public_release.py
python scripts/render_readme_assets.py
```

Before a release, run the installation doctor, tests, lint, and public-release
boundary check on a clean Python 3.11 or 3.12 environment. The repository is
ready for Streamlit Community Cloud, but deployment requires authorization
from the repository owner; this README does not advertise an unverified hosted
URL.

## Repository layout

```text
streamlit_app.py          browser application
src/rootfpt/              simulation, metrics, hydraulics, and app helpers
configs/                  explicit synthetic presets and example selectors
tests/                    analytical, stochastic, conservation, and UI tests
tutorials/                small executable examples
workflows/                command-line research workflows
scripts/                  installation, release, and documentation checks
assets/                   reproducible software-facing README images
```

This public repository contains software and software-facing documentation
only. Research-writing and submission materials remain outside its release
boundary.

## Contributing, support, and citation

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Report
reproducible defects through [GitHub Issues](https://github.com/nalin-dhiman/Plantroot/issues)
with the seed, settings, platform, Python version, and commit hash.

The software is available under the [MIT License](LICENSE). Citation metadata
is provided in [CITATION.cff](CITATION.cff).
