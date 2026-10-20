"""NephroTwin dashboard.

Run locally with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nephrotwin.clinical import gfr_category, gfr_category_description
from nephrotwin.config import (
    EGFR_CKD_THRESHOLD,
    EGFR_KIDNEY_FAILURE,
    EGFR_SEVERE_THRESHOLD,
    FIGURES_DIR,
    METRICS_FILE,
    PATIENTS_FILE,
    PRIOR_FILE,
    VISITS_FILE,
)
from nephrotwin.explain import explain
from nephrotwin.interventions import CATALOGUE, simulate
from nephrotwin.summary import patient_facts
from nephrotwin.twin import DigitalTwin, PopulationPrior

GREEN = "#1F6F43"
GREY = "#8A949E"
AMBER = "#D08C2E"
RED = "#C0392B"

st.set_page_config(page_title="NephroTwin | Kidney Digital Twin", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_artifacts():
    visits = pd.read_csv(VISITS_FILE, parse_dates=["date"])
    patients = pd.read_csv(PATIENTS_FILE)
    metrics = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
    prior_dict = json.loads(PRIOR_FILE.read_text())
    return visits, patients, metrics, prior_dict


try:
    VISITS, PATIENTS, METRICS, PRIOR_DICT = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artefacts were not found. Build them with "
        "`python -m nephrotwin.pipeline --synthea <path to Synthea output/csv>` and reload."
    )
    st.stop()

PRIOR = PopulationPrior.from_dict(PRIOR_DICT)
COVARIATES = list(PRIOR.covariate_names)

if "care_plans" not in st.session_state:
    st.session_state.care_plans = []


def patient_covariates(row: pd.Series) -> dict:
    return {c: (0.0 if pd.isna(row.get(c)) else float(row.get(c))) for c in COVARIATES}


def build_twin(pid: str, k: int) -> tuple[DigitalTwin, pd.DataFrame, pd.Series]:
    row = PATIENTS.set_index("patient_id").loc[pid]
    sub = VISITS[VISITS["patient_id"] == pid].sort_values("t_years").reset_index(drop=True)
    twin = DigitalTwin(PRIOR, patient_covariates(row), pid)
    twin.assimilate_many(sub["t_years"].to_numpy()[:k], sub["egfr"].to_numpy()[:k])
    return twin, sub, row


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("NephroTwin")
st.sidebar.caption("A personal digital twin of kidney function")

role_options = {"Held-out test patients (never seen in training)": "test", "All patients": None}
role_label = st.sidebar.radio("Patient group", list(role_options), index=0)
pool = PATIENTS if role_options[role_label] is None else PATIENTS[PATIENTS["cohort_role"] == role_options[role_label]]

stage_filter = st.sidebar.multiselect(
    "Starting kidney function (KDIGO)", ["G1", "G2", "G3a", "G3b", "G4"], default=[]
)
if stage_filter:
    pool = pool[pool["baseline_egfr"].apply(gfr_category).isin(stage_filter)]

order = st.sidebar.selectbox("Sort patients by", ["Largest decline first", "Most results first", "Patient ID"])
pool = pool.assign(decline=pool["baseline_egfr"] - pool["last_egfr"])
if order == "Largest decline first":
    pool = pool.sort_values("decline", ascending=False)
elif order == "Most results first":
    pool = pool.sort_values("n_visits", ascending=False)
else:
    pool = pool.sort_values("patient_id")

if pool.empty:
    st.warning("No patients match the current filters.")
    st.stop()


def describe(pid: str) -> str:
    r = PATIENTS.set_index("patient_id").loc[pid]
    return f"{pid} · {r['sex']} · age {r['age_at_entry']:.0f} · eGFR {r['baseline_egfr']:.0f}→{r['last_egfr']:.0f}"


pid = st.sidebar.selectbox("Patient", pool["patient_id"].tolist(), format_func=describe)
n_total = int(PATIENTS.set_index("patient_id").loc[pid, "n_visits"])
k = st.sidebar.slider(
    "Results the twin has seen",
    min_value=1,
    max_value=n_total,
    value=max(1, min(n_total, max(3, n_total // 2))),
    help="Replay history: move the slider to feed the twin one laboratory result at a time.",
)
horizon = st.sidebar.slider("Forecast horizon (years)", 1, 10, 5)
st.sidebar.divider()
st.sidebar.caption("Decision support only. Not a medical device. Synthetic data.")

twin, sub, row = build_twin(pid, k)
summary = twin.state_summary()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("NephroTwin")
st.caption("Personal kidney-function digital twin · decision support, not medical advice")

tab_twin, tab_whatif, tab_explain, tab_perf, tab_about = st.tabs(
    ["Patient twin", "What-if simulator", "Explanation", "Model performance", "About the model"]
)

# ---------------------------------------------------------------------------
# Tab 1: patient twin
# ---------------------------------------------------------------------------
with tab_twin:
    g_cat = gfr_category(summary["current_egfr_estimate"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Estimated eGFR now", f"{summary['current_egfr_estimate']:.0f}", f"± {summary['current_egfr_sd']:.0f}", delta_color="off")
    c2.metric("KDIGO category", g_cat, gfr_category_description(g_cat), delta_color="off")
    c3.metric(
        "Change per year",
        f"{summary['slope_per_year']:+.1f}",
        f"80%: {summary['slope_lower']:+.1f} to {summary['slope_upper']:+.1f}",
        delta_color="off",
    )
    p30 = twin.prob_below(EGFR_SEVERE_THRESHOLD, twin.t_last + horizon)
    c4.metric(f"P(eGFR < 30 in {horizon} y)", f"{100 * p30:.0f}%")
    c5.metric("Twin confidence", summary["confidence"].split(":")[0], f"{k} of {n_total} results", delta_color="off")

    grid = np.linspace(twin.t_last, twin.t_last + horizon, 80)
    fc = twin.predict(grid)
    hist = pd.DataFrame(
        [{"t": s.t, "level": s.level, "lo": s.level - 1.28 * s.level_sd, "hi": s.level + 1.28 * s.level_sd} for s in twin.history]
    )
    seen, unseen = sub.iloc[:k], sub.iloc[k:]
    flagged = [s for s in twin.history if s.down_weighted]

    fig = go.Figure()
    for y, color, label in ((EGFR_CKD_THRESHOLD, GREY, "60: CKD threshold"), (EGFR_SEVERE_THRESHOLD, AMBER, "30: severe"), (EGFR_KIDNEY_FAILURE, RED, "15: kidney failure")):
        fig.add_hline(y=y, line_dash="dot", line_color=color, line_width=1, annotation_text=label, annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=hist["t"], y=hist["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist["t"], y=hist["lo"], fill="tonexty", fillcolor="rgba(31,111,67,0.10)", line=dict(width=0), name="Twin belief (80%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist["t"], y=hist["level"], line=dict(color=GREEN, width=2), name="Twin estimate"))
    fig.add_trace(go.Scatter(x=fc["t"], y=fc["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["t"], y=fc["lower"], fill="tonexty", fillcolor="rgba(31,111,67,0.18)", line=dict(width=0), name="Forecast range (80%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["t"], y=fc["mean"], line=dict(color=GREEN, width=2, dash="dash"), name="Forecast"))
    fig.add_trace(go.Scatter(x=seen["t_years"], y=seen["egfr"], mode="markers", marker=dict(color="#222", size=7), name="Results seen by twin",
                             customdata=seen["date"].dt.strftime("%Y-%m-%d"), hovertemplate="%{customdata}<br>eGFR %{y:.0f}<extra></extra>"))
    if not unseen.empty:
        fig.add_trace(go.Scatter(x=unseen["t_years"], y=unseen["egfr"], mode="markers", marker=dict(color="rgba(0,0,0,0)", size=7, line=dict(color=GREY, width=1.5)),
                                 name="Actual later results (hidden from twin)", hovertemplate="eGFR %{y:.0f}<extra></extra>"))
    if flagged:
        fig.add_trace(go.Scatter(x=[s.t for s in flagged], y=[s.observed_egfr for s in flagged], mode="markers",
                                 marker=dict(symbol="x", size=11, color=RED), name="Atypical result (down-weighted)"))

    for plan in st.session_state.care_plans:
        if plan["patient_id"] == pid and plan["k"] < k:
            fig.add_trace(go.Scatter(x=plan["t"], y=plan["median"], line=dict(color=AMBER, width=2, dash="dot"),
                                     name=f"Expected under plan recorded at result {plan['k']}"))

    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Years since first result",
                      yaxis_title="eGFR (mL/min/1.73m²)", legend=dict(orientation="h", y=-0.2), template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    if flagged:
        st.info(f"{len(flagged)} result(s) were unusually far from the twin's expectation and were given less weight. "
                "In practice this would prompt a repeat test.")

    ttf = twin.time_to_threshold(EGFR_KIDNEY_FAILURE, horizon_years=10, n_draws=3000)
    a, b, c = st.columns(3)
    a.metric("P(eGFR < 15 within 10 y)", f"{100 * ttf['prob_within_horizon']:.1f}%")
    b.metric("P(rapid decline > 5 / yr)", f"{100 * summary['prob_rapid_decline']:.1f}%")
    facts_now = patient_facts(twin, row, sub, horizon_years=horizon)
    if facts_now.get("kfre_5yr") is not None:
        c.metric("KFRE 5-year kidney-failure risk", f"{100 * facts_now['kfre_5yr']:.1f}%", "valid range" if facts_now["kfre_applicable"] else "outside validated range", delta_color="off")
    else:
        c.metric("KFRE 5-year risk", "n/a", "needs urine ACR", delta_color="off")

    with st.expander("How the twin's belief changed after each result"):
        table = pd.DataFrame(
            [{"result #": s.n_observed, "years": round(s.t, 2), "measured eGFR": round(s.observed_egfr, 1),
              "twin eGFR": round(s.level, 1), "twin slope / yr": round(s.slope, 2), "slope uncertainty (sd)": round(s.slope_sd, 2),
              "surprise (z)": round(s.surprise, 2), "down-weighted": s.down_weighted} for s in twin.history]
        )
        st.dataframe(table, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2: what-if
# ---------------------------------------------------------------------------
with tab_whatif:
    st.subheader("Simulate a care plan")
    st.caption("Treatment effects are drawn from published trial ranges and act on the twin's personal rate of decline.")
    left, right = st.columns([1, 2])
    with left:
        keys = st.multiselect("Interventions", list(CATALOGUE), format_func=lambda key: CATALOGUE[key].name, default=["sglt2i"])
        delay = st.slider("Start in (years)", 0.0, 2.0, 0.0, 0.25)
        years = st.slider("Simulate (years)", 1, 10, horizon)
    scen = simulate(twin, keys, years_ahead=years, start_offset_years=delay, n_draws=3000)
    with right:
        f2 = go.Figure()
        for name, band, color, fill in (("Usual care", scen["usual_care"], GREY, "rgba(138,148,158,0.18)"),
                                        ("With plan", scen["scenario"], GREEN, "rgba(31,111,67,0.18)")):
            f2.add_trace(go.Scatter(x=band["t"], y=band["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            f2.add_trace(go.Scatter(x=band["t"], y=band["lower"], fill="tonexty", fillcolor=fill, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            f2.add_trace(go.Scatter(x=band["t"], y=band["median"], line=dict(color=color, width=2.5), name=name))
        f2.add_hline(y=EGFR_KIDNEY_FAILURE, line_dash="dot", line_color=RED)
        f2.add_hline(y=EGFR_SEVERE_THRESHOLD, line_dash="dot", line_color=AMBER)
        f2.update_layout(height=380, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title="Years since first result", yaxis_title="eGFR", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(f2, width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"eGFR difference at {years} y", f"{scen['egfr_gain_median']:+.1f}", f"80%: {scen['egfr_gain_lower']:+.1f} to {scen['egfr_gain_upper']:+.1f}", delta_color="off")
    m2.metric(f"P(eGFR < 15 within {years} y), usual care", f"{100 * scen['prob_kidney_failure_usual']:.1f}%")
    m3.metric("With plan", f"{100 * scen['prob_kidney_failure_scenario']:.1f}%")

    if keys:
        st.markdown("**Evidence behind the selected scenarios**")
        st.dataframe(
            pd.DataFrame([{"Intervention": CATALOGUE[x].name, "Slower decline (range)": f"{CATALOGUE[x].low:.0%} to {CATALOGUE[x].high:.0%}",
                           "Most likely": f"{CATALOGUE[x].mode:.0%}", "Evidence": CATALOGUE[x].evidence,
                           "Applies when": CATALOGUE[x].applies_when, "Sources": CATALOGUE[x].references} for x in keys]),
            width="stretch", hide_index=True,
        )
        if len(keys) > 1:
            st.caption("Combined effects assume independent mechanisms; this is a modelling assumption, not trial evidence.")

    if st.button("Record this plan for the patient", disabled=not keys):
        st.session_state.care_plans.append({
            "patient_id": pid, "k": k, "interventions": [CATALOGUE[x].name for x in keys],
            "t": scen["scenario"]["t"].tolist(), "median": scen["scenario"]["median"].tolist(),
            "recorded": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        st.success("Plan recorded. Move the 'Results the twin has seen' slider forward to compare later results with the plan's expectation.")

    plans = [p for p in st.session_state.care_plans if p["patient_id"] == pid]
    if plans:
        st.markdown("**Care plan log**")
        st.dataframe(pd.DataFrame([{"Recorded": p["recorded"], "After result #": p["k"], "Plan": ", ".join(p["interventions"])} for p in plans]),
                     hide_index=True, width="stretch")
    st.warning("Scenarios illustrate average trial effects applied to this twin. They are not personal predictions of treatment response and must not guide prescribing.")

# ---------------------------------------------------------------------------
# Tab 3: explanation
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("Plain-language summary")
    include_scenario = st.checkbox("Include the current what-if scenario", value=bool(keys))
    facts = patient_facts(twin, row, sub, scenario=scen if include_scenario and keys else None, horizon_years=horizon)
    if st.button("Generate summary", type="primary"):
        with st.spinner("Writing summary..."):
            result = explain(facts)
        st.markdown(result.text)
        st.caption("Written by " + ("a language model (" + result.source.split(":", 1)[1] + ") from the facts below." if result.source.startswith("llm") else "a fixed template (no language-model key configured)."))
    with st.expander("Facts supplied to the summary writer"):
        st.json(facts)

# ---------------------------------------------------------------------------
# Tab 4: performance
# ---------------------------------------------------------------------------
with tab_perf:
    if not METRICS:
        st.info("No validation report found.")
    else:
        val = METRICS["validation"]
        st.subheader("Validation on held-out patients")
        st.caption(
            f"{val['split']['test_patients']} test patients never used for fitting · "
            f"{val['split']['test_forecasts_next_visit']} next-result forecasts · "
            f"{val['split']['test_forecasts_one_year_plus']} forecasts one year or more ahead"
        )
        names = {"twin": "NephroTwin", "last_value": "Last value carried forward", "patient_mean": "Patient mean so far",
                 "patient_ols": "Straight line through own results", "population": "Population average only"}
        rows = []
        for key, label in names.items():
            rows.append({"Method": label, "Next result MAE": round(val["next_visit"][key]["mae"], 2),
                         "1 year+ MAE": round(val["one_year_plus"][key]["mae"], 2) if val.get("one_year_plus") else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        pc = val.get("paired_comparisons", {})
        if pc:
            st.markdown("**Paired comparisons (patient-level bootstrap, 95% CI; negative favours NephroTwin)**")
            st.dataframe(pd.DataFrame([{"Comparison": k.replace("_", " "), "MAE difference": round(v["mae_difference"], 2),
                                        "95% CI": f"{v['ci95_lower']:.2f} to {v['ci95_upper']:.2f}"} for k, v in pc.items()]),
                         hide_index=True, width="stretch")

        cal = val["interval_calibration"]
        e = val["event_prediction"]
        x1, x2, x3 = st.columns(3)
        x1.metric("80% interval coverage (next result)", f"{100 * cal['coverage_next_visit']:.0f}%")
        if cal.get("coverage_one_year_plus") is not None:
            x2.metric("80% interval coverage (1 year+)", f"{100 * cal['coverage_one_year_plus']:.0f}%")
        if "twin_auroc" in e:
            x3.metric("Event AUROC (eGFR < 30 at ~2 y)", f"{e['twin_auroc']:.2f}", f"{e['n_events']} events, preliminary", delta_color="off")
        else:
            x3.metric("Event prediction", "n/a", e.get("note", ""), delta_color="off")

        for fname, caption in (("learning_curve.png", "Error falls as the twin sees more of a patient's results."),
                               ("interval_calibration.png", "Share of real results inside the twin's 80% range."),
                               ("method_comparison.png", "Forecast error by method."),
                               ("example_twin.png", "One held-out patient: the forecast tightens as results arrive.")):
            path = FIGURES_DIR / fname
            if path.exists():
                st.image(str(path), caption=caption)

        with st.expander("Cohort and data-quality report"):
            st.json({"cohort": METRICS.get("cohort"), "ingestion": METRICS.get("ingestion"), "kfre": METRICS.get("kfre")})

# ---------------------------------------------------------------------------
# Tab 5: about
# ---------------------------------------------------------------------------
with tab_about:
    pm = METRICS.get("population_model", {})
    st.subheader("What this twin is")
    st.markdown(
        """
**NephroTwin** keeps a living statistical model of one person's kidney function. Each new creatinine result
updates the twin's belief about the person's current eGFR and how fast it is changing. From that belief it forecasts
future kidney function with an honest uncertainty range, estimates the chance of crossing clinical thresholds, simulates
care plans, and explains the result in plain language.

**How it works**
- eGFR from creatinine with the race-free CKD-EPI 2021 equation; KDIGO 2024 staging; 4-variable KFRE where valid.
- Bayesian state-space model (local linear trend Kalman filter) with a hierarchical population prior fitted by EM.
- Prior structure chosen by cross-validation; dynamics tuned on predictive accuracy; intervals calibrated by split-conformal prediction.
- Robust updating: an unusually surprising single result is down-weighted and flagged for repeat testing.

**Intended use**: research and education; demonstrating decision support for CKD monitoring.
**Not intended for**: diagnosis, prescribing, or use on real patients without clinical validation and regulatory approval.

**Main limitations**: trained on synthetic data (Synthea), which is mostly stable kidney function with few progression
events; treatment effects are trial averages, not individual responses; no validation yet on Indian patient data.
"""
    )
    if pm:
        st.json({k: pm[k] for k in ("measurement_sd", "between_patient_sd_level", "between_patient_sd_slope",
                                    "process_noise_sd_level_per_year", "process_noise_sd_slope_per_year", "conformal_interval_scale") if k in pm})
    st.caption(f"Data: {METRICS.get('data_source', '')} · {METRICS.get('data_provenance', '')} · built {METRICS.get('generated_utc', '')}")
