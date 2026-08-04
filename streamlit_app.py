"""Stable Streamlit Community Cloud entry point for ROOT-FPT Explorer."""

from pathlib import Path
from runpy import run_path

# Streamlit re-executes this file after every widget event. ``run_path`` also
# re-executes the UI module; a normal import would incorrectly reuse its cache.
run_path(str(Path(__file__).resolve().parent / "app" / "main.py"), run_name="__main__")
