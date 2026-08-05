# Streamlit deployment

Deploy the application from the repository root with these settings:

- Repository: `nalin-dhiman/Plantroot`
- Branch: `main`
- Entry point: `streamlit_app.py`
- Python: `3.12`

Use `requirements.txt` as the dependency file. If the deployment log reports
Python 3.13 or newer, delete and recreate the app with Python 3.12 selected in
Advanced settings; rebooting does not change an existing app's Python runtime.

For unrestricted public use, set the Streamlit sharing option to **This app is
public and searchable**. A redirect to Streamlit sign-in means the deployment
is still private.

The repository-level entry point handles Streamlit reruns and removes stale
ROOT-FPT modules after a versioned deployment update. Application code lives
in `app/main.py`; simulation code remains in `src/rootfpt/`.
