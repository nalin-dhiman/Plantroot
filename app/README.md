# Browser application

`main.py` contains the Streamlit user interface. The repository-level
`streamlit_app.py` is a deliberately small compatibility entry point for
Streamlit Community Cloud and existing local commands.

Run the application from the repository root so configuration and software
assets resolve consistently:

```bash
python -m streamlit run streamlit_app.py
```

The interface calls the public `rootfpt.explorer` API. Simulation and analysis
logic belongs in `src/rootfpt/`; UI-only state and presentation belong here.
