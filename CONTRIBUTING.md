# Contributing

Contributions that improve correctness, testing, documentation, performance,
or accessibility are welcome.

## Before opening a change

1. Open an issue for changes to equations, parameter semantics, random-number
   coupling, units, or output schemas.
2. Keep synthetic presets clearly separated from empirical calibration.
3. Add or update a test for every behavior change.
4. Preserve deterministic results for unchanged seeds unless the change is an
   intentional numerical correction documented in the pull request.

## Local checks

```bash
python -m pip install -r requirements-dev.txt
pytest -q
ruff check .
python scripts/check_public_release.py
```

For Streamlit changes, also run:

```bash
streamlit run streamlit_app.py
```

Include the seed and exact settings for visual or stochastic defects. Avoid
claims of biological realism unless they are supported by an explicit
calibration and validation contribution.

