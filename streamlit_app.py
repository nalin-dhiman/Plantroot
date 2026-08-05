"""Stable Streamlit Community Cloud entry point for ROOT-FPT Explorer."""

from pathlib import Path
from runpy import run_path

from app.bootstrap import discard_stale_rootfpt_modules

# Streamlit re-executes this file after every widget event. ``run_path`` also
# re-executes the UI module; a normal import would incorrectly reuse its cache.
REPOSITORY_ROOT = Path(__file__).resolve().parent
discard_stale_rootfpt_modules(REPOSITORY_ROOT)
run_path(str(REPOSITORY_ROOT / "app" / "main.py"), run_name="__main__")
