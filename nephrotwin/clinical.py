"""Validated clinical equations used by the twin.

References
----------
CKD-EPI 2021
    Inker LA, Eneanya ND, Coresh J, et al. New creatinine- and cystatin C-based
    equations to estimate GFR without race. N Engl J Med. 2021;385:1737-1749.
KDIGO staging
    KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of
    Chronic Kidney Disease. Kidney Int. 2024;105(4S):S117-S314.
Kidney Failure Risk Equation (4-variable)
    Tangri N, Stevens LA, Griffith J, et al. A predictive model for progression of
    chronic kidney disease to kidney failure. JAMA. 2011;305(15):1553-1559.
    Tangri N, Grams ME, Levey AS, et al. Multinational assessment of accuracy of
    equations for predicting risk of kidney failure. JAMA. 2016;315(2):164-174.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ArrayLike = float | np.ndarray | pd.Series


# ---------------------------------------------------------------------------
# eGFR: CKD-EPI 2021 creatinine equation (race-free)
# ---------------------------------------------------------------------------
def _is_female(sex: ArrayLike) -> np.ndarray:
    values = np.atleast_1d(np.asarray(sex, dtype=object))
    return np.array([str(v).strip().upper().startswith("F") for v in values])


def egfr_ckd_epi_2021(creatinine_mg_dl: ArrayLike, age_years: ArrayLike, sex: ArrayLike) -> np.ndarray:
    """Estimate GFR (mL/min/1.73m2) with the 2021 race-free CKD-EPI creatinine equation.

    eGFR = 142 * min(Scr/k, 1)^a * max(Scr/k, 1)^-1.200 * 0.9938^age * 1.012 [if female]
    with k = 0.7 (female) or 0.9 (male) and a = -0.241 (female) or -0.302 (male).

    Parameters
    ----------
    creatinine_mg_dl : serum creatinine in mg/dL (must be positive).
    age_years : age in years (equation validated for adults, age >= 18).
    sex : "M"/"F" (any string starting with F is treated as female).

    Returns
    -------
    numpy array of eGFR values. Non-positive or missing creatinine yields NaN.
    """
    scr = np.atleast_1d(np.asarray(creatinine_mg_dl, dtype=float))
    age = np.atleast_1d(np.asarray(age_years, dtype=float))
    female = _is_female(sex)
    scr, age, female = np.broadcast_arrays(scr, age, female)

    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    sex_factor = np.where(female, 1.012, 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = scr / kappa
        egfr = (
            142.0
            * np.power(np.minimum(ratio, 1.0), alpha)
            * np.power(np.maximum(ratio, 1.0), -1.200)
            * np.power(0.9938, age)
            * sex_factor
        )
    egfr = np.where((scr > 0) & np.isfinite(scr) & np.isfinite(age), egfr, np.nan)
    return egfr


# ---------------------------------------------------------------------------
# KDIGO staging
# ---------------------------------------------------------------------------
_G_BOUNDS = [(90.0, "G1"), (60.0, "G2"), (45.0, "G3a"), (30.0, "G3b"), (15.0, "G4"), (-np.inf, "G5")]
_G_LABELS = {
    "G1": "Normal or high",
    "G2": "Mildly decreased",
    "G3a": "Mildly to moderately decreased",
    "G3b": "Moderately to severely decreased",
    "G4": "Severely decreased",
    "G5": "Kidney failure",
}


def gfr_category(egfr: float) -> str:
    """Return the KDIGO GFR category (G1 to G5) for a single eGFR value."""
    if egfr is None or not np.isfinite(egfr):
        return "Unknown"
    for bound, label in _G_BOUNDS:
        if egfr >= bound:
            return label
    return "G5"


def gfr_category_description(category: str) -> str:
    return _G_LABELS.get(category, "Unknown")


def albuminuria_category(acr_mg_g: float | None) -> str:
    """Return the KDIGO albuminuria category (A1 to A3) for a urine ACR in mg/g."""
    if acr_mg_g is None or not np.isfinite(acr_mg_g):
        return "Unknown"
    if acr_mg_g < 30:
        return "A1"
    if acr_mg_g <= 300:
        return "A2"
    return "A3"


def kdigo_risk_band(g_cat: str, a_cat: str) -> str:
    """Map a G/A combination to the KDIGO heat-map risk band.

    Returns one of: "Low", "Moderately increased", "High", "Very high", "Unknown".
    """
    grid = {
        "G1": ("Low", "Moderately increased", "High"),
        "G2": ("Low", "Moderately increased", "High"),
        "G3a": ("Moderately increased", "High", "Very high"),
        "G3b": ("High", "Very high", "Very high"),
        "G4": ("Very high", "Very high", "Very high"),
        "G5": ("Very high", "Very high", "Very high"),
    }
    if g_cat not in grid:
        return "Unknown"
    if a_cat not in {"A1", "A2", "A3"}:
        # Without albuminuria the GFR category alone sets a lower bound on risk.
        return grid[g_cat][0]
    return grid[g_cat][{"A1": 0, "A2": 1, "A3": 2}[a_cat]]


# ---------------------------------------------------------------------------
# Kidney Failure Risk Equation (4-variable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KfreCalibration:
    """Baseline survival values for the 4-variable KFRE."""

    name: str
    s0_2yr: float
    s0_5yr: float


KFRE_ORIGINAL = KfreCalibration("Original (Tangri 2011)", s0_2yr=0.9750, s0_5yr=0.9240)
KFRE_NON_NORTH_AMERICA = KfreCalibration("Non-North American recalibration (Tangri 2016)", s0_2yr=0.9832, s0_5yr=0.9365)

_KFRE_BETA_AGE = -0.2201
_KFRE_BETA_MALE = 0.2467
_KFRE_BETA_EGFR = -0.5567
_KFRE_BETA_LNACR = 0.4510
_KFRE_MEAN_AGE10 = 7.036
_KFRE_MEAN_MALE = 0.5642
_KFRE_MEAN_EGFR5 = 7.222
_KFRE_MEAN_LNACR = 5.137


@dataclass(frozen=True)
class KfreResult:
    risk_2yr: float
    risk_5yr: float
    applicable: bool
    note: str
    calibration: str


def kfre_4var(
    age_years: float,
    sex: str,
    egfr: float,
    acr_mg_g: float,
    calibration: KfreCalibration = KFRE_NON_NORTH_AMERICA,
) -> KfreResult:
    """4-variable Kidney Failure Risk Equation.

    risk(t) = 1 - S0(t) ^ exp(LP), where
    LP = -0.2201*(age/10 - 7.036) + 0.2467*(male - 0.5642)
         - 0.5567*(eGFR/5 - 7.222) + 0.4510*(ln(ACR) - 5.137)

    The equation is validated for adults with CKD G3 to G5 (eGFR < 60).
    Outside that range the risk is still computed but ``applicable`` is False.
    """
    if any(v is None or not np.isfinite(v) for v in (age_years, egfr, acr_mg_g)) or acr_mg_g <= 0:
        return KfreResult(np.nan, np.nan, False, "Requires age, eGFR and a positive urine ACR.", calibration.name)

    male = 0.0 if _is_female(sex)[0] else 1.0
    lp = (
        _KFRE_BETA_AGE * (age_years / 10.0 - _KFRE_MEAN_AGE10)
        + _KFRE_BETA_MALE * (male - _KFRE_MEAN_MALE)
        + _KFRE_BETA_EGFR * (egfr / 5.0 - _KFRE_MEAN_EGFR5)
        + _KFRE_BETA_LNACR * (np.log(acr_mg_g) - _KFRE_MEAN_LNACR)
    )
    hazard_multiplier = float(np.exp(lp))
    risk_2 = 1.0 - calibration.s0_2yr ** hazard_multiplier
    risk_5 = 1.0 - calibration.s0_5yr ** hazard_multiplier

    applicable = egfr < 60.0 and age_years >= 18
    note = (
        "Validated range (adult, eGFR < 60)."
        if applicable
        else "Outside validated range (eGFR >= 60 or age < 18); interpret with caution."
    )
    return KfreResult(float(risk_2), float(risk_5), applicable, note, calibration.name)
