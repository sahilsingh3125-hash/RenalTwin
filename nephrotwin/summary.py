"""Assemble the clinical facts that describe one patient's twin at a point in time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clinical import (
    KFRE_NON_NORTH_AMERICA,
    albuminuria_category,
    gfr_category,
    gfr_category_description,
    kdigo_risk_band,
    kfre_4var,
)
from .config import EGFR_KIDNEY_FAILURE, EGFR_SEVERE_THRESHOLD
from .twin import DigitalTwin


def latest_value(visits: pd.DataFrame, column: str, up_to_t: float | None = None) -> float | None:
    if column not in visits:
        return None
    sub = visits if up_to_t is None else visits[visits["t_years"] <= up_to_t + 1e-9]
    sub = sub.dropna(subset=[column])
    return None if sub.empty else float(sub[column].iloc[-1])


def patient_facts(
    twin: DigitalTwin,
    patient: pd.Series | dict,
    visits: pd.DataFrame,
    scenario: dict | None = None,
    horizon_years: float = 5.0,
) -> dict:
    """Return a JSON-serialisable dictionary describing the twin's current state."""
    patient = dict(patient)
    state = twin.state_summary()
    t_now = twin.t_last
    egfr_now = state["current_egfr_estimate"]
    acr = latest_value(visits, "acr", t_now)
    sbp = latest_value(visits, "sbp", t_now)
    hba1c = latest_value(visits, "hba1c", t_now)

    age_now = float(patient.get("age_at_entry", np.nan)) + t_now
    g_cat = gfr_category(egfr_now)
    a_cat = albuminuria_category(acr)

    facts: dict = {
        **state,
        "age": round(age_now, 1) if np.isfinite(age_now) else None,
        "sex": patient.get("sex"),
        "diabetes": bool(patient.get("diabetes", 0)),
        "hypertension": bool(patient.get("hypertension", 0)),
        "latest_urine_acr_mg_g": acr,
        "latest_systolic_bp": sbp,
        "latest_hba1c": hba1c,
        "kdigo_gfr_category": g_cat,
        "kdigo_gfr_description": gfr_category_description(g_cat),
        "kdigo_albuminuria_category": a_cat,
        "kdigo_risk_band": kdigo_risk_band(g_cat, a_cat),
        "atypical_results": int(sum(s.down_weighted for s in twin.history)),
    }

    ttf = twin.time_to_threshold(EGFR_KIDNEY_FAILURE, horizon_years=10.0, n_draws=3000)
    facts["prob_egfr_below_15_within_10y"] = ttf["prob_within_horizon"]
    facts[f"prob_egfr_below_30_in_{int(horizon_years)}y"] = twin.prob_below(EGFR_SEVERE_THRESHOLD, t_now + horizon_years)

    if acr is not None and np.isfinite(age_now):
        k = kfre_4var(age_now, str(patient.get("sex", "")), egfr_now, acr, KFRE_NON_NORTH_AMERICA)
        facts.update({"kfre_2yr": k.risk_2yr, "kfre_5yr": k.risk_5yr, "kfre_applicable": k.applicable, "kfre_note": k.note})

    if scenario:
        facts["scenario"] = {
            "interventions": scenario.get("interventions", []),
            "years_ahead": scenario.get("years_ahead"),
            "egfr_gain_median": scenario.get("egfr_gain_median"),
            "egfr_gain_lower": scenario.get("egfr_gain_lower"),
            "egfr_gain_upper": scenario.get("egfr_gain_upper"),
            "prob_kidney_failure_usual": scenario.get("prob_kidney_failure_usual"),
            "prob_kidney_failure_scenario": scenario.get("prob_kidney_failure_scenario"),
        }

    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in facts.items()}
