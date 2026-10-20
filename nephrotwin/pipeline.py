"""End-to-end build of every NephroTwin artefact.

Usage
-----
    python -m nephrotwin.pipeline --synthea data/raw/synthea/csv
    python -m nephrotwin.pipeline --synthea batch1/csv batch2/csv
    python -m nephrotwin.pipeline --table my_cohort.csv --time-col t_years --egfr-col egfr

Outputs
-------
    data/processed/visits.csv           one row per patient per day of testing
    data/processed/patients.csv         one row per patient with baseline facts
    models/population_prior.json        fitted population model
    reports/metrics.json                validation results and data provenance
    reports/learning_curve.csv          error by number of results seen
    reports/event_predictions.csv       landmark event predictions
    reports/figures/*.png               report figures
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, plots
from .clinical import KFRE_NON_NORTH_AMERICA, kfre_4var
from .config import (
    EVENT_PREDICTIONS_FILE,
    FIGURES_DIR,
    LEARNING_CURVE_FILE,
    METRICS_FILE,
    PATIENTS_FILE,
    PRIOR_FILE,
    SETTINGS,
    VISITS_FILE,
)
from .evaluation import evaluate, split_patients
from .ingest import load_longitudinal_table, load_synthea, to_series

log = logging.getLogger("nephrotwin")


def kfre_assessment(patients: pd.DataFrame, horizon_years: float = 2.0) -> dict:
    """Apply the KFRE at cohort entry where it is valid and compare with observed kidney failure."""
    eligible = patients[(patients["baseline_egfr"] < 60) & patients["baseline_acr"].notna() & (patients["baseline_acr"] > 0)]
    rows = []
    for _, r in eligible.iterrows():
        k = kfre_4var(r["age_at_entry"], r["sex"], r["baseline_egfr"], r["baseline_acr"], KFRE_NON_NORTH_AMERICA)
        observed = bool(r["kidney_failure_event"]) and r["years_to_kidney_failure"] <= horizon_years
        followed = r["followup_years"] >= horizon_years or observed
        rows.append({"risk_2yr": k.risk_2yr, "risk_5yr": k.risk_5yr, "event_2yr": observed, "followed_2yr": followed})
    df = pd.DataFrame(rows)
    out = {
        "eligible_patients": int(len(eligible)),
        "calibration": KFRE_NON_NORTH_AMERICA.name,
        "note": "KFRE is applied at cohort entry to adults with eGFR < 60 and a urine ACR result.",
    }
    if not df.empty:
        evaluable = df[df["followed_2yr"]]
        out.update(
            {
                "mean_predicted_2yr_risk": float(df["risk_2yr"].mean()),
                "evaluable_2yr": int(len(evaluable)),
                "observed_kidney_failure_2yr": int(evaluable["event_2yr"].sum()),
            }
        )
    return out


def cohort_description(visits: pd.DataFrame, patients: pd.DataFrame) -> dict:
    def pct(col: str) -> float:
        return float(100 * patients[col].fillna(0).mean())

    stages = pd.cut(
        patients["baseline_egfr"], [-np.inf, 15, 30, 45, 60, 90, np.inf], labels=["G5", "G4", "G3b", "G3a", "G2", "G1"], right=False
    ).value_counts().reindex(["G1", "G2", "G3a", "G3b", "G4", "G5"]).fillna(0).astype(int)
    return {
        "patients": int(len(patients)),
        "laboratory_results": int(len(visits)),
        "median_results_per_patient": float(patients["n_visits"].median()),
        "median_followup_years": float(patients["followup_years"].median()),
        "median_age_at_entry": float(patients["age_at_entry"].median()),
        "female_percent": float(100 * (patients["sex"] == "F").mean()),
        "diabetes_percent": pct("diabetes"),
        "hypertension_percent": pct("hypertension"),
        "ckd_g3_or_worse_at_entry_percent": float(100 * (patients["baseline_egfr"] < 60).mean()),
        "kidney_failure_events": int(patients["kidney_failure_event"].sum()),
        "baseline_kdigo_gfr_categories": {k: int(v) for k, v in stages.items()},
    }


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    settings = SETTINGS
    for d in (VISITS_FILE.parent, PRIOR_FILE.parent, METRICS_FILE.parent, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion -----------------------------------------------------------------
    if args.table:
        table = pd.read_csv(args.table)
        visits, patients, report = load_longitudinal_table(
            table,
            patient_col=args.patient_col,
            date_col=args.date_col,
            time_years_col=args.time_col,
            egfr_col=args.egfr_col,
            creatinine_col=args.creatinine_col,
            age_col=args.age_col,
            sex_col=args.sex_col,
            acr_col=args.acr_col,
            settings=settings,
        )
    else:
        visits, patients, report = load_synthea(args.synthea, settings)
    log.info("Cohort: %d patients, %d results", len(patients), len(visits))

    # 2. Validation (fitting / calibration / test split) ------------------------------
    series = to_series(visits, patients, settings.model.prior_covariates)
    prior, result = evaluate(series, settings.model)
    prior.save(PRIOR_FILE)

    fit, cal, test = split_patients(
        series, settings.model.test_fraction, settings.model.calibration_fraction, settings.model.random_seed
    )
    role = {s.patient_id: "fitting" for s in fit} | {s.patient_id: "calibration" for s in cal} | {
        s.patient_id: "test" for s in test
    }
    patients["cohort_role"] = patients["patient_id"].map(role)

    # 3. Persist tables ---------------------------------------------------------------
    visits_out = visits.copy()
    visits_out["date"] = visits_out["date"].dt.date
    num_cols = visits_out.select_dtypes("number").columns
    visits_out[num_cols] = visits_out[num_cols].round(3)
    visits_out.to_csv(VISITS_FILE, index=False)

    patients_out = patients.copy()
    for col in ("entry_date", "last_date", "kidney_failure_date"):
        patients_out[col] = pd.to_datetime(patients_out[col]).dt.date
    num_cols = patients_out.select_dtypes("number").columns
    patients_out[num_cols] = patients_out[num_cols].round(3)
    patients_out.to_csv(PATIENTS_FILE, index=False)

    result.learning_curve.round(4).to_csv(LEARNING_CURVE_FILE, index=False)
    result.events.round(4).to_csv(EVENT_PREDICTIONS_FILE, index=False)

    # 4. Figures ------------------------------------------------------------------------
    plots.learning_curve(result.learning_curve, FIGURES_DIR / "learning_curve.png")
    plots.method_comparison(result.metrics, FIGURES_DIR / "method_comparison.png")
    plots.interval_calibration(result.learning_curve, settings.model.interval_level, FIGURES_DIR / "interval_calibration.png")
    test_sorted = sorted(test, key=lambda s: (s.egfr[0] - s.egfr[-1]) * min(s.n, 15), reverse=True)
    if test_sorted:
        plots.example_twin(prior, test_sorted[0], FIGURES_DIR / "example_twin.png")

    # 5. Metrics and provenance ---------------------------------------------------------------
    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nephrotwin_version": __version__,
        "data_source": report.source,
        "data_provenance": args.provenance or "",
        "ingestion": report.to_dict(),
        "cohort": cohort_description(visits, patients),
        "population_model": prior.summary(),
        "validation": result.metrics,
        "kfre": kfre_assessment(patients),
        "runtime_seconds": round(time.time() - started, 1),
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the NephroTwin model, validation report and app data.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthea", nargs="+", help="One or more Synthea output/csv directories.")
    src.add_argument("--table", help="A tidy longitudinal CSV (one row per measurement).")
    p.add_argument("--patient-col", default="patient_id")
    p.add_argument("--date-col", default=None)
    p.add_argument("--time-col", default=None, help="Time since baseline in years (alternative to a date).")
    p.add_argument("--egfr-col", default=None)
    p.add_argument("--creatinine-col", default=None)
    p.add_argument("--age-col", default="age")
    p.add_argument("--sex-col", default="sex")
    p.add_argument("--acr-col", default=None)
    p.add_argument("--provenance", default="", help="Free-text description of how the data were produced.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    metrics = run(args)
    v = metrics["validation"]
    nv = v["next_visit"]
    print(f"Cohort: {metrics['cohort']['patients']} patients, {metrics['cohort']['laboratory_results']} results")
    print(f"Test patients: {v['split']['test_patients']}  (forecasts scored: {v['split']['test_forecasts_next_visit']})")
    print("Next-result mean absolute error (mL/min/1.73m2):")
    for k, m in nv.items():
        print(f"  {k:<14} {m['mae']:6.2f}")
    cal = v["interval_calibration"]
    print(f"80% interval coverage on test patients: {100 * cal['coverage_next_visit']:.1f}%")
    print(f"Artefacts written to {METRICS_FILE.parent.parent}")


if __name__ == "__main__":
    main()
