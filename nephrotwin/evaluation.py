"""Validation of the twin on patients it was not fitted on.

Design
------
* Patients are split into fitting, calibration and test cohorts. The population
  model is fitted and tuned on the fitting cohort, prediction intervals are
  calibrated on the calibration cohort, and every reported number comes from
  the test cohort, which the model never sees during development.
* Rolling-origin forecasting: for every test patient and every number of
  observed visits k, the twin sees visits 1..k and forecasts visit k+1 (and all
  visits at least one year ahead). This mirrors real use, where the twin is
  updated after each new laboratory result.
* Four comparators are scored on the same forecasts:
    last_value      the most recent eGFR carried forward (common clinical habit)
    patient_mean    the average of the patient's results so far
    patient_ols     a straight line through the patient's results so far
    population      the population prior alone, with no personal data
* Interval calibration: the share of real measurements that fall inside the
  twin's 80% prediction interval should be close to 80%.
* Event prediction (landmark analysis): from a landmark visit with eGFR above a
  threshold, predict whether the measured eGFR about two years later is below
  that threshold. Scored by AUROC and Brier score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import ModelSettings
from .twin import DigitalTwin, PatientSeries, PopulationPrior


def split_patients(
    series: list[PatientSeries], test_fraction: float, calibration_fraction: float, seed: int
) -> tuple[list, list, list]:
    """Patient-level split into fitting, calibration and test cohorts."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(s.patient_id for s in series))
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_fraction)))
    n_cal = max(1, int(round(len(ids) * calibration_fraction)))
    test_ids = set(ids[:n_test])
    cal_ids = set(ids[n_test : n_test + n_cal])
    fit = [s for s in series if s.patient_id not in test_ids and s.patient_id not in cal_ids]
    cal = [s for s in series if s.patient_id in cal_ids]
    test = [s for s in series if s.patient_id in test_ids]
    return fit, cal, test


def select_structure(
    fit: list[PatientSeries],
    settings: ModelSettings,
    ridge_grid=(0.0, 10.0, 50.0, 250.0),
    independence_grid=(False, True),
    n_folds: int = 3,
) -> dict:
    """Choose prior shrinkage and level-slope dependence by cross-validation.

    Only the fitting cohort is used. Each candidate is fitted on all but one
    fold and scored on the held-out fold by the average of next-result and
    one-year-ahead absolute error. The test cohort is never touched.
    """
    rng = np.random.default_rng(settings.random_seed + 1)
    ids = np.array(sorted(s.patient_id for s in fit))
    rng.shuffle(ids)
    fold_of = {pid: i % n_folds for i, pid in enumerate(ids)}
    table = []
    for ridge in ridge_grid:
        for independent in independence_grid:
            scores = []
            for f in range(n_folds):
                train = [s for s in fit if fold_of[s.patient_id] != f]
                held = [s for s in fit if fold_of[s.patient_id] == f]
                try:
                    pr = PopulationPrior.fit(
                        train, settings.prior_covariates, settings.em_max_iter, settings.em_tol, ridge, independent
                    )
                except ValueError:
                    continue
                scores.append(pr.predictive_log_likelihood(held)["combined_mae"])
            table.append({"ridge": ridge, "independent_effects": independent, "cv_mae": float(np.mean(scores))})
    best = min(table, key=lambda r: r["cv_mae"])
    return {"selected": best, "candidates": table, "folds": n_folds}


def build_prior(fit: list[PatientSeries], cal: list[PatientSeries], settings: ModelSettings) -> PopulationPrior:
    """Select structure by cross-validation, fit, tune dynamics and calibrate intervals."""
    structure = select_structure(fit, settings)
    chosen = structure["selected"]
    prior = PopulationPrior.fit(
        fit,
        settings.prior_covariates,
        settings.em_max_iter,
        settings.em_tol,
        ridge=chosen["ridge"],
        independent_effects=chosen["independent_effects"],
    )
    prior.tune_dynamics(fit)
    prior.tuning["structure_selection"] = structure
    prior.calibrate_intervals(cal, settings.interval_level)
    return prior


def _ols_predict(t_hist: np.ndarray, y_hist: np.ndarray, t_new: float) -> float:
    if t_hist.size >= 2 and np.ptp(t_hist) > 0.05:
        slope, intercept = np.polyfit(t_hist, y_hist, 1)
        return float(intercept + slope * t_new)
    return float(y_hist[-1])


@dataclass
class EvaluationResult:
    forecasts: pd.DataFrame
    events: pd.DataFrame
    metrics: dict
    learning_curve: pd.DataFrame


def rolling_forecasts(prior: PopulationPrior, test: list[PatientSeries], level: float) -> pd.DataFrame:
    """Forecast each next visit (and visits >= 1 year ahead) from every origin."""
    rows = []
    for s in test:
        twin = DigitalTwin(prior, s.covariates, s.patient_id)
        pop_mean = prior.prior_mean(s.covariates)
        for k in range(1, s.n):
            twin.assimilate(s.t[k - 1], s.egfr[k - 1])
            t_hist, y_hist = s.t[:k], s.egfr[:k]
            targets = [k] + [j for j in range(k + 1, s.n) if s.t[j] - s.t[k - 1] >= 1.0]
            for j in targets:
                t_new, y_true = s.t[j], s.egfr[j]
                pred = twin.predict(t_new, level=level, include_noise=True).iloc[0]
                rows.append(
                    {
                        "patient_id": s.patient_id,
                        "k_observed": k,
                        "horizon_years": t_new - s.t[k - 1],
                        "next_visit": j == k,
                        "actual": y_true,
                        "twin": pred["mean"],
                        "twin_lower": pred["lower"],
                        "twin_upper": pred["upper"],
                        "last_value": y_hist[-1],
                        "patient_mean": float(np.mean(y_hist)),
                        "patient_ols": _ols_predict(t_hist, y_hist, t_new),
                        "population": float(pop_mean[0] + pop_mean[1] * t_new),
                    }
                )
    return pd.DataFrame(rows)


def landmark_events(
    prior: PopulationPrior,
    test: list[PatientSeries],
    threshold: float,
    horizon: float,
    window: float = 0.5,
) -> pd.DataFrame:
    """Predict eGFR below threshold about ``horizon`` years after each landmark."""
    rows = []
    for s in test:
        twin = DigitalTwin(prior, s.covariates, s.patient_id)
        last_landmark_t = -np.inf
        for k in range(1, s.n):
            twin.assimilate(s.t[k - 1], s.egfr[k - 1])
            t_land = s.t[k - 1]
            if k < 2 or s.egfr[k - 1] < threshold or t_land - last_landmark_t < 1.0:
                continue
            future = np.where(np.abs(s.t - (t_land + horizon)) <= window)[0]
            if future.size == 0:
                continue
            j = future[np.argmin(np.abs(s.t[future] - (t_land + horizon)))]
            pred = twin.predict(s.t[j], include_noise=True).iloc[0]
            sd = max(pred["sd"], 1e-6)
            p_twin = float(stats.norm.cdf(threshold, loc=pred["mean"], scale=sd))

            pop = prior.prior_mean(s.covariates)
            pop_mean = pop[0] + pop[1] * s.t[j]
            pop_sd = float(np.sqrt(np.array([1, s.t[j]]) @ prior.Sigma @ np.array([1, s.t[j]]) + prior.sigma2))
            p_pop = float(stats.norm.cdf(threshold, loc=pop_mean, scale=pop_sd))

            # Last value with measurement noise only (no trend information).
            p_last = float(stats.norm.cdf(threshold, loc=s.egfr[k - 1], scale=np.sqrt(2 * prior.sigma2)))

            rows.append(
                {
                    "patient_id": s.patient_id,
                    "landmark_t": t_land,
                    "landmark_egfr": s.egfr[k - 1],
                    "k_observed": k,
                    "outcome_t": s.t[j],
                    "outcome_egfr": s.egfr[j],
                    "event": int(s.egfr[j] < threshold),
                    "p_twin": p_twin,
                    "p_population": p_pop,
                    "p_last_value": p_last,
                }
            )
            last_landmark_t = t_land
    return pd.DataFrame(rows)


def _error_metrics(df: pd.DataFrame, methods: list[str]) -> dict:
    out = {}
    for m in methods:
        err = df[m] - df["actual"]
        out[m] = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2))), "bias": float(np.mean(err))}
    return out


def bootstrap_difference(df: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Patient-level bootstrap of the difference in mean absolute error (a minus b).

    Resampling whole patients respects the correlation between forecasts made
    for the same person. A negative difference means method ``a`` is more accurate.
    """
    if df.empty:
        return {}
    per = df.assign(ea=(df[a] - df["actual"]).abs(), eb=(df[b] - df["actual"]).abs()).groupby("patient_id")[["ea", "eb"]].agg(["sum", "count"])
    sa, sb, n = per[("ea", "sum")].to_numpy(), per[("eb", "sum")].to_numpy(), per[("ea", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    diffs = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    point = float((sa.sum() - sb.sum()) / n.sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "comparison": f"{a} minus {b}",
        "mae_difference": point,
        "ci95_lower": float(lo),
        "ci95_upper": float(hi),
        "share_of_bootstraps_favouring_" + a: float(np.mean(diffs < 0)),
    }


def _event_metrics(events: pd.DataFrame) -> dict:
    if events.empty:
        return {"n_landmarks": 0, "n_events": 0, "note": "No eligible landmarks."}
    n_events = int(events["event"].sum())
    result: dict = {
        "n_landmarks": int(len(events)),
        "n_patients": int(events["patient_id"].nunique()),
        "n_events": n_events,
        "event_rate": float(events["event"].mean()),
    }
    if n_events >= 5 and n_events <= len(events) - 5:
        for col in ("p_twin", "p_population", "p_last_value"):
            result[col.replace("p_", "") + "_auroc"] = float(roc_auc_score(events["event"], events[col]))
            result[col.replace("p_", "") + "_brier"] = float(brier_score_loss(events["event"], events[col]))
    else:
        result["note"] = "Fewer than five events or non-events; discrimination metrics not reported."
    return result


def evaluate(series: list[PatientSeries], settings: ModelSettings) -> tuple[PopulationPrior, EvaluationResult]:
    """Fit on the fitting split and score every method on the held-out split."""
    fit, cal, test = split_patients(series, settings.test_fraction, settings.calibration_fraction, settings.random_seed)
    prior = build_prior(fit, cal, settings)

    fc = rolling_forecasts(prior, test, settings.interval_level)
    methods = ["twin", "last_value", "patient_mean", "patient_ols", "population"]
    next_visit = fc[fc["next_visit"]]
    long_h = fc[fc["horizon_years"] >= settings.long_horizon_years]

    inside = (next_visit["actual"] >= next_visit["twin_lower"]) & (next_visit["actual"] <= next_visit["twin_upper"])
    inside_long = (long_h["actual"] >= long_h["twin_lower"]) & (long_h["actual"] <= long_h["twin_upper"])

    curve_rows = []
    grouped = next_visit.assign(k=next_visit["k_observed"].clip(upper=12)).groupby("k")
    for k, g in grouped:
        curve_rows.append(
            {
                "k": int(k),
                "n": int(len(g)),
                "twin_mae": float(np.mean(np.abs(g["twin"] - g["actual"]))),
                "last_value_mae": float(np.mean(np.abs(g["last_value"] - g["actual"]))),
                "patient_ols_mae": float(np.mean(np.abs(g["patient_ols"] - g["actual"]))),
                "population_mae": float(np.mean(np.abs(g["population"] - g["actual"]))),
                "interval_width": float(np.mean(g["twin_upper"] - g["twin_lower"])),
                "coverage": float(np.mean((g["actual"] >= g["twin_lower"]) & (g["actual"] <= g["twin_upper"]))),
            }
        )
    curve = pd.DataFrame(curve_rows)

    events = landmark_events(prior, test, settings.event_threshold, settings.event_horizon_years)

    metrics = {
        "split": {
            "fitting_patients": len(fit),
            "calibration_patients": len(cal),
            "test_patients": len(test),
            "test_forecasts_next_visit": int(len(next_visit)),
            "test_forecasts_one_year_plus": int(len(long_h)),
        },
        "next_visit": _error_metrics(next_visit, methods),
        "one_year_plus": _error_metrics(long_h, methods) if len(long_h) else {},
        "paired_comparisons": {
            "next_visit_vs_last_value": bootstrap_difference(next_visit, "twin", "last_value"),
            "next_visit_vs_patient_ols": bootstrap_difference(next_visit, "twin", "patient_ols"),
            "one_year_plus_vs_last_value": bootstrap_difference(long_h, "twin", "last_value"),
            "one_year_plus_vs_patient_ols": bootstrap_difference(long_h, "twin", "patient_ols"),
        },
        "interval_calibration": {
            "nominal_level": settings.interval_level,
            "coverage_next_visit": float(inside.mean()),
            "coverage_one_year_plus": float(inside_long.mean()) if len(long_h) else None,
            "mean_width_next_visit": float((next_visit["twin_upper"] - next_visit["twin_lower"]).mean()),
        },
        "event_prediction": {
            "definition": (
                f"Measured eGFR below {settings.event_threshold:g} about "
                f"{settings.event_horizon_years:g} years after a landmark visit with eGFR at or above it."
            ),
            **_event_metrics(events),
        },
    }
    return prior, EvaluationResult(fc, events, metrics, curve)
