"""ROOT-FPT Explorer: a bounded interface for synthetic root--soil experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import rootfpt.explorer as explorer_api
from rootfpt import __version__
from rootfpt.explorer import (
    labels,
    load_default_config,
    relative_change,
    result_archive,
    result_signature,
    run_experiment,
)
from rootfpt.visualization.explorer import (
    architecture_figure,
    comparison_figure,
    root_length_density_figure,
)

ATLAS_DURATION_DAYS = float(getattr(explorer_api, "ATLAS_DURATION_DAYS", 5.5))
MAX_EXPLORER_DURATION_DAYS = float(
    getattr(explorer_api, "MAX_EXPLORER_DURATION_DAYS", ATLAS_DURATION_DAYS)
)

st.set_page_config(
    page_title="ROOT-FPT Explorer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink: #172033; --leaf: #16865c; --sand: #f3eadc; }
      [data-testid="stAppViewContainer"] { background: #fbfcfa; }
      [data-testid="stSidebar"] { background: #f2f6f1; }
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1450px; }
      .hero {
        padding: 1.35rem 1.5rem; border: 1px solid #dce6dc; border-radius: 16px;
        background: linear-gradient(120deg, #f3f8f3 0%, #fffaf2 100%);
        margin-bottom: 1rem;
      }
      .hero h1 { color: var(--ink); margin: 0; font-size: 2.15rem; letter-spacing: -0.025em; }
      .hero p { color: #465266; margin: 0.45rem 0 0; max-width: 850px; }
      .scope-note {
        padding: .8rem 1rem; border-left: 4px solid #d97706; background: #fff8e8;
        border-radius: 6px; color: #574515; margin: .4rem 0 1rem;
      }
      [data-testid="stMetric"] {
        border: 1px solid #e1e8e1; padding: .72rem;
        border-radius: 12px; background: white;
      }
      .smallcaps {
        color: #64748b; font-size: .78rem; letter-spacing: .08em;
        text-transform: uppercase; font-weight: 700;
      }
      div[data-testid="stDownloadButton"] button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=1800, max_entries=6)
def cached_experiment(
    architecture: str,
    soil: str,
    seed: int,
    replicate: int,
    duration: float,
    dt: float,
    max_tips: int,
):
    return run_experiment(
        architecture,
        soil,
        seed=seed,
        replicate=replicate,
        duration_days=duration,
        dt_days=dt,
        max_tips=max_tips,
    )


config = load_default_config()
architecture_labels = labels(config, "root_regimes")
soil_labels = labels(config, "soils")
architecture_keys = list(architecture_labels)
soil_keys = list(soil_labels)

st.markdown(
    """
    <div class="hero">
      <div class="smallcaps">Open synthetic root–soil laboratory</div>
      <h1>ROOT-FPT Explorer</h1>
      <p>Grow a stochastic branching architecture, compare paired virtual roots,
      inspect numerical sensitivity, and download analysis-ready outputs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="scope-note"><strong>Scope:</strong> This is a two-dimensional,
    synthetic and uncalibrated research model. Preset names are parameter labels,
    not plant species. Outputs are not field predictions or management advice.</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Experiment controls")
    st.caption("Inputs are explicit and every run is reproducible from its seed.")
    with st.form("experiment_controls"):
        selected_architecture = st.selectbox(
            "Architecture preset",
            architecture_keys,
            index=2,
            format_func=architecture_labels.get,
        )
        selected_soil = st.selectbox(
            "Soil constructor",
            soil_keys,
            index=1,
            format_func=soil_labels.get,
        )
        seed = int(
            st.number_input(
                "Master seed",
                min_value=0,
                max_value=2_147_483_647,
                value=20260802,
                step=1,
            )
        )
        replicate = int(st.number_input("Replicate index", 0, 9999, 0, 1))
        horizon_options = [f"Atlas window · up to {ATLAS_DURATION_DAYS:g} d"]
        if MAX_EXPLORER_DURATION_DAYS > ATLAS_DURATION_DAYS:
            horizon_options.append(
                f"Extended preview · up to {MAX_EXPLORER_DURATION_DAYS:g} d"
            )
        horizon = st.radio(
            "Development horizon",
            horizon_options,
            help=(
                "Extended runs enlarge the synthetic soil domain but extrapolate "
                "the short-window model without adding ageing or turnover."
            ),
        )
        extended_horizon = horizon.startswith("Extended")
        if extended_horizon:
            duration = float(
                st.select_slider(
                    "Duration (days)",
                    options=(7, 14, 21, 30),
                    value=14,
                    format_func=lambda value: (
                        "30 days · about 1 month" if value == 30 else f"{value} days"
                    ),
                    key="extended_duration",
                )
            )
            dt = 0.04
            st.caption("Extended previews use the 0.04-day integration step.")
            max_tips = int(
                st.slider(
                    "Maximum total-tip allocation",
                    40,
                    60,
                    60,
                    10,
                    key="extended_tip_cap",
                    help="Capped at 60 to protect the shared application worker.",
                )
            )
        else:
            duration = float(
                st.slider(
                    "Duration (days)",
                    1.0,
                    ATLAS_DURATION_DAYS,
                    ATLAS_DURATION_DAYS,
                    0.5,
                    key="atlas_duration",
                )
            )
            resolution = st.radio(
                "Time resolution",
                ("Standard · 0.04 d", "Fine · 0.02 d"),
                horizontal=False,
                help="Fine mode approximately doubles integration work.",
            )
            dt = 0.02 if resolution.startswith("Fine") else 0.04
            max_tips = int(
                st.slider(
                    "Maximum total-tip allocation",
                    40,
                    160,
                    120,
                    10,
                    key="atlas_tip_cap",
                    help="A computational guardrail, not a biological carrying capacity.",
                )
            )
        run_clicked = st.form_submit_button("Grow this root", type="primary", width="stretch")

    if "active_parameters" not in st.session_state:
        st.session_state.active_parameters = {
            "architecture": "dimorphic",
            "soil": "patchy_matern",
            "seed": 20260802,
            "replicate": 0,
            "duration": 5.5,
            "dt": 0.04,
            "max_tips": 120,
        }
    if run_clicked:
        st.session_state.active_parameters = {
            "architecture": selected_architecture,
            "soil": selected_soil,
            "seed": seed,
            "replicate": replicate,
            "duration": duration,
            "dt": dt,
            "max_tips": max_tips,
        }
    st.divider()
    st.caption(
        "Tip: change only the replicate index to explore stochastic variation "
        "without changing parameters."
    )

parameters = st.session_state.active_parameters
extended_run = parameters["duration"] > ATLAS_DURATION_DAYS
if extended_run:
    st.warning(
        "Extended preview: all extended horizons use one enlarged month-scale soil "
        "canvas, but switching from the atlas window starts a new expanded-domain run; "
        "it does not continue the displayed 5.5-day root. The same short-window growth "
        "rates are extrapolated without root ageing, turnover, seasonal forcing, or "
        "dynamic carbon feedback, so this is not a validated month-scale prediction."
    )
with st.spinner("Integrating root development and terminal hydraulics…"):
    result = cached_experiment(**parameters)

explore_tab, compare_tab, diagnostics_tab, methods_tab = st.tabs(
    ["Explore", "Paired comparison", "Diagnostics", "Model & limits"]
)

with explore_tab:
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        figure = architecture_figure(
            result,
            architecture_labels[result.architecture_name],
            soil_labels[result.soil_name],
        )
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        st.caption(
            "Blue shading is the prescribed water field. Brown contours mark high impedance; "
            "green dashed contours mark anisotropic channels when present."
        )
    with right:
        st.subheader("Run summary")
        first, second = st.columns(2)
        first.metric("Maximum depth", f"{result.metrics['maximum_depth_cm']:.2f} cm")
        second.metric("Total length", f"{result.metrics['total_root_length_cm']:.2f} cm")
        first.metric("Horizontal spread", f"{result.metrics['horizontal_spread_cm']:.2f} cm")
        second.metric("Branches", f"{int(result.metrics['branch_count'])}")
        first.metric("Segments", f"{int(result.metrics['segment_count'])}")
        second.metric("Mean tortuosity", f"{result.metrics['mean_tortuosity']:.2f}")
        st.caption(
            "The hydraulic index is calculated on the final architecture; "
            "it is not dynamically fed back into growth."
        )
        hydraulic, carbon = st.columns(2)
        hydraulic.metric(
            "Hydraulic index",
            f"{result.metrics['cumulative_hydraulic_index']:.3g}",
        )
        carbon.metric("Construction cost", f"{result.metrics['construction_cost']:.2f}")
        if result.metrics["construction_budget_exceeded"]:
            st.warning(
                "This exploratory run exceeded the shared construction-accounting reference."
            )
        if result.metrics.get("tip_allocation_reached", 0):
            st.warning(
                "The total-tip allocation was reached. Later potential branches were "
                "not allocated, so this architecture is computationally truncated."
            )
        if result.metrics.get("boundary_contact_count", 0):
            st.warning(
                f"{int(result.metrics['boundary_contact_count'])} tip(s) reached the "
                "synthetic soil boundary and stopped. Interpret size-dependent metrics "
                "with that truncation in mind."
            )

        rld_figure = root_length_density_figure(result)
        st.pyplot(rld_figure, width="stretch")
        plt.close(rld_figure)

        export_name = (
            f"rootfpt_{result.architecture_name}_{result.soil_name}_"
            f"seed{parameters['seed']}_rep{parameters['replicate']}.zip"
        )
        st.download_button(
            "Download metrics, segments, and depth profile",
            data=result_archive(result),
            file_name=export_name,
            mime="application/zip",
            width="stretch",
        )
        with st.expander("Exact run record"):
            st.code(
                json.dumps(
                    {
                        **result.settings,
                        "replicate": result.replicate,
                        "architecture": result.architecture_name,
                        "soil": result.soil_name,
                        **result.seeds,
                        "segment_sha256": result_signature(result),
                    },
                    indent=2,
                ),
                language="json",
            )

with compare_tab:
    st.subheader("Paired virtual-root comparison")
    st.write(
        "Two presets share each soil realization at a given replicate index. "
        "This controls environmental randomness, but a small interactive sample is exploratory."
    )
    if extended_run:
        st.info(
            "Paired ensembles are limited to the 5.5-day atlas window in the shared "
            "web app. Use the Python workflow for long-horizon ensembles."
        )
    with st.form("comparison_controls"):
        control_columns = st.columns(4)
        architecture_a = control_columns[0].selectbox(
            "Preset A", architecture_keys, index=0, format_func=architecture_labels.get
        )
        architecture_b = control_columns[1].selectbox(
            "Preset B", architecture_keys, index=1, format_func=architecture_labels.get
        )
        comparison_soil = control_columns[2].selectbox(
            "Shared soil", soil_keys, index=1, format_func=soil_labels.get
        )
        comparison_count = int(control_columns[3].slider("Paired replicates", 3, 8, 5))
        compare_clicked = st.form_submit_button(
            "Run paired comparison",
            disabled=extended_run,
        )

    if compare_clicked:
        if architecture_a == architecture_b:
            st.warning("Choose two different architecture presets for a useful comparison.")
        else:
            rows: list[dict] = []
            progress = st.progress(0, text="Running paired realizations…")
            total = 2 * comparison_count
            completed = 0
            for replicate_index in range(comparison_count):
                for code, architecture_name in (("a", architecture_a), ("b", architecture_b)):
                    paired = cached_experiment(
                        architecture_name,
                        comparison_soil,
                        parameters["seed"],
                        replicate_index,
                        parameters["duration"],
                        parameters["dt"],
                        parameters["max_tips"],
                    )
                    rows.append(
                        {
                            "architecture": code,
                            "replicate": replicate_index,
                            **paired.metrics,
                            "environment_seed": paired.seeds["environment_seed"],
                        }
                    )
                    completed += 1
                    progress.progress(completed / total, text="Running paired realizations…")
            progress.empty()
            comparison = pd.DataFrame(rows)
            paired_seed_counts = comparison.groupby("replicate")["environment_seed"].nunique()
            if not bool((paired_seed_counts == 1).all()):
                st.error("Environment pairing failed; comparison suppressed.")
            else:
                comparison_plot = comparison_figure(
                    comparison,
                    architecture_labels[architecture_a],
                    architecture_labels[architecture_b],
                )
                st.pyplot(comparison_plot, width="stretch")
                plt.close(comparison_plot)
                summaries = (
                    comparison.groupby("architecture")
                    .agg(
                        depth_mean_cm=("maximum_depth_cm", "mean"),
                        length_mean_cm=("total_root_length_cm", "mean"),
                        branches_mean=("branch_count", "mean"),
                    )
                    .rename(
                        index={
                            "a": architecture_labels[architecture_a],
                            "b": architecture_labels[architecture_b],
                        }
                    )
                )
                st.dataframe(summaries.style.format("{:.3f}"), width="stretch")
                st.caption(
                    "Lines connect identical environment seeds. No confidence interval "
                    "is claimed for this small sample."
                )

with diagnostics_tab:
    st.subheader("Numerical and accounting diagnostics")
    st.write(
        "The check below reruns the active condition at 0.04 and 0.02 day using "
        "event-keyed randomness and the same resolved Brownian path. A single run "
        "is a smoke test, not a global convergence proof."
    )
    if extended_run:
        st.info(
            "Interactive 0.04/0.02-day resolution checks are limited to the 5.5-day "
            "atlas window because a month-scale paired solve is too costly for the "
            "shared worker."
        )
    if st.button("Run resolution check", key="resolution_check", disabled=extended_run):
        with st.spinner("Running coupled coarse and fine integrations…"):
            coarse = cached_experiment(
                parameters["architecture"],
                parameters["soil"],
                parameters["seed"],
                parameters["replicate"],
                parameters["duration"],
                0.04,
                parameters["max_tips"],
            )
            fine = cached_experiment(
                parameters["architecture"],
                parameters["soil"],
                parameters["seed"],
                parameters["replicate"],
                parameters["duration"],
                0.02,
                parameters["max_tips"],
            )
        diagnostic_metrics = {
            "Maximum depth": "maximum_depth_cm",
            "Horizontal spread": "horizontal_spread_cm",
            "Total root length": "total_root_length_cm",
            "Branch count": "branch_count",
        }
        rows = []
        for label, key in diagnostic_metrics.items():
            change = relative_change(float(coarse.metrics[key]), float(fine.metrics[key]))
            rows.append(
                {
                    "Metric": label,
                    "0.04 d": coarse.metrics[key],
                    "0.02 d": fine.metrics[key],
                    "Relative change (%)": change,
                    "≤ 5% reference": change <= 5.0,
                }
            )
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {"0.04 d": "{:.4g}", "0.02 d": "{:.4g}", "Relative change (%)": "{:.2f}"}
            ),
            width="stretch",
            hide_index=True,
        )
        if all(row["≤ 5% reference"] for row in rows):
            st.success("This sentinel run is within the 5% reference for the displayed metrics.")
        else:
            st.warning(
                "At least one displayed metric exceeds the 5% reference. Use finer "
                "settings and an ensemble before interpretation."
            )

        accounting = pd.DataFrame(
            [
                {
                    "Check": "Hydraulic Kirchhoff residual",
                    "Observed": fine.metrics["kirchhoff_residual"],
                    "Reference": 1e-8,
                    "Pass": fine.metrics["kirchhoff_residual"] <= 1e-8,
                },
                {
                    "Check": "Construction-balance residual",
                    "Observed": fine.metrics["construction_balance_residual"],
                    "Reference": 1e-10,
                    "Pass": fine.metrics["construction_balance_residual"] <= 1e-10,
                },
            ]
        )
        st.dataframe(
            accounting.style.format({"Observed": "{:.3e}", "Reference": "{:.1e}"}),
            hide_index=True,
            width="stretch",
        )

    st.markdown("#### Exact reproducibility")
    st.code(result_signature(result), language="text")
    st.caption(
        "This SHA-256 identifies the ordered segment table for the active seed and settings."
    )

with methods_tab:
    st.subheader("What the software does")
    overview, image_column = st.columns([1.05, 1], gap="large")
    with overview:
        st.markdown(
            """
            - Evolves persistent active tips with directional soil responses and angular noise.
            - Creates developmentally delayed lateral sites without splitting the parent axis.
            - Builds explicit graph topology and reports geometry and depth profiles.
            - Solves a terminal axial–radial hydraulic network on the final graph.
            - Uses named deterministic seeds for repeatability and paired comparisons.

            **Important limits**

            - Two-dimensional effective model; roots do not occupy a three-dimensional volume.
            - Soil fields are controlled constructors, not measured profiles.
            - The reduced water and terminal hydraulic calculations do not feed
              back into development.
            - No species calibration, parameter inference, rhizosphere
              resolution, or field validation.
            - Extended previews enlarge the domain but do not add ageing,
              turnover, seasonal forcing, or dynamic carbon limitation.
            - Presets should not be ranked as biologically superior from these outputs.
            """
        )
    with image_column:
        framework = REPOSITORY_ROOT / "assets" / "model_framework.png"
        if framework.exists():
            st.image(
                str(framework),
                caption="Model components and visual encoding",
                width="stretch",
            )
    st.info(
        "For research use, record the complete run settings, seed, software commit, "
        "and downloaded segment hash. "
        "Use ensembles and numerical checks for claims that depend on stochastic variability."
    )

st.divider()
st.caption(
    f"ROOT-FPT Explorer {__version__} · © 2026 Nalin Dhiman, IIT Mandi · "
    "MIT-licensed · Synthetic outputs only"
)
