"""Central configuration for NephroTwin.

All clinical codes, thresholds and file locations live here so that every other
module reads from a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

VISITS_FILE = PROCESSED_DIR / "visits.csv"
PATIENTS_FILE = PROCESSED_DIR / "patients.csv"
PRIOR_FILE = MODELS_DIR / "population_prior.json"
METRICS_FILE = REPORTS_DIR / "metrics.json"
LEARNING_CURVE_FILE = REPORTS_DIR / "learning_curve.csv"
EVENT_PREDICTIONS_FILE = REPORTS_DIR / "event_predictions.csv"


# ---------------------------------------------------------------------------
# Laboratory and vital-sign codes (LOINC)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoincCodes:
    """LOINC codes used to extract kidney-relevant observations."""

    creatinine: tuple[str, ...] = ("2160-0", "38483-4")   # serum/plasma, blood (mg/dL)
    acr: tuple[str, ...] = ("14959-1", "9318-7")          # urine albumin/creatinine (mg/g)
    systolic_bp: tuple[str, ...] = ("8480-6",)            # mm[Hg]
    diastolic_bp: tuple[str, ...] = ("8462-4",)           # mm[Hg]
    hba1c: tuple[str, ...] = ("4548-4",)                  # %
    bmi: tuple[str, ...] = ("39156-5",)                   # kg/m2


# ---------------------------------------------------------------------------
# Diagnosis codes (SNOMED CT)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SnomedCodes:
    """SNOMED CT concepts used to derive comorbidities and outcomes."""

    diabetes: tuple[str, ...] = ("44054006",)
    prediabetes: tuple[str, ...] = ("714628002", "15777000")
    hypertension: tuple[str, ...] = ("59621000", "38341003")
    ckd_any: tuple[str, ...] = (
        "431855005",  # CKD stage 1
        "431856006",  # CKD stage 2
        "433144002",  # CKD stage 3
        "431857002",  # CKD stage 4
        "433146000",  # CKD stage 5
        "46177005",   # End-stage renal disease
        "709044004",  # Chronic kidney disease (generic)
        "127013003",  # Diabetic renal disease
        "90781000119102",  # Microalbuminuria due to type 2 diabetes
    )
    kidney_failure: tuple[str, ...] = (
        "46177005",   # End-stage renal disease
        "433146000",  # CKD stage 5
    )
    kidney_failure_text: tuple[str, ...] = (
        "end-stage renal disease",
        "end stage renal disease",
        "chronic kidney disease stage 5",
        "dependence on renal dialysis",
        "kidney transplant",
    )


# ---------------------------------------------------------------------------
# Clinical thresholds (KDIGO 2024)
# ---------------------------------------------------------------------------
EGFR_CKD_THRESHOLD = 60.0          # mL/min/1.73m2, CKD G3a and below
EGFR_SEVERE_THRESHOLD = 30.0       # CKD G4 and below
EGFR_KIDNEY_FAILURE = 15.0         # CKD G5
RAPID_DECLINE_PER_YEAR = -5.0      # KDIGO definition of rapid progression


# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------
@dataclass
class ModelSettings:
    """Hyper-parameters for cohort selection, the twin and evaluation."""

    min_age: float = 18.0
    min_visits: int = 4
    min_followup_years: float = 1.0
    egfr_floor: float = 3.0
    egfr_ceiling: float = 150.0
    same_day_merge: bool = True

    em_max_iter: int = 200
    em_tol: float = 1e-6
    prior_covariates: tuple[str, ...] = ("age10", "diabetes", "hypertension", "male")

    interval_level: float = 0.80
    forecast_years: float = 5.0
    monte_carlo_draws: int = 4000

    test_fraction: float = 0.30
    calibration_fraction: float = 0.15
    random_seed: int = 42
    event_threshold: float = EGFR_SEVERE_THRESHOLD
    event_horizon_years: float = 2.0
    long_horizon_years: float = 1.0


@dataclass
class Settings:
    """Top-level settings object passed through the pipeline."""

    loinc: LoincCodes = field(default_factory=LoincCodes)
    snomed: SnomedCodes = field(default_factory=SnomedCodes)
    model: ModelSettings = field(default_factory=ModelSettings)


SETTINGS = Settings()
