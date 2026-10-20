"""Build a clean longitudinal kidney-function timeline from raw records.

Two entry points are provided:

``load_synthea``
    Reads Synthea CSV exports (patients, observations, conditions, procedures)
    and produces one row per patient per day with eGFR and linked measurements.

``load_longitudinal_table``
    Accepts any tidy longitudinal table (for example an external research
    cohort) with a patient identifier, a date or time column and either eGFR or
    creatinine with age and sex.

Data-quality rules applied to every source
------------------------------------------
1. Only creatinine measured in outpatient settings (wellness, ambulatory,
   outpatient visits) is used. Values from inpatient, emergency and urgent-care
   encounters often reflect acute kidney injury rather than chronic kidney
   function, and KDIGO defines CKD by chronicity (more than three months).
   Same-day values are reduced to one value per day; the serum or plasma assay
   is preferred over whole-blood when both exist.
2. Internal consistency: when the same encounter reports both a creatinine and
   a laboratory-calculated eGFR, the two must agree. A creatinine whose implied
   eGFR differs from the reported eGFR by more than 40% (beyond the expected
   MDRD versus CKD-EPI difference) is treated as unreliable and removed.
3. Follow-up of the native kidney stops at kidney failure (first ESRD or CKD G5
   diagnosis, dialysis or kidney transplant). Values recorded after kidney
   replacement therapy do not describe native kidney function and are removed.
   The kidney failure date is kept as an outcome.
4. Physiologically implausible eGFR values are removed, and isolated spikes are
   removed with a rolling-median filter on the log scale.
5. Only pseudonymous identifiers, sex and dates needed for modelling are kept.
   Names, addresses and identity numbers present in the source are never read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .clinical import egfr_ckd_epi_2021
from .config import SETTINGS, Settings

log = logging.getLogger(__name__)

DIALYSIS_CODES = ("265764009",)
TRANSPLANT_CODES = ("70536003", "161665007")
OUTPATIENT_CLASSES = ("wellness", "ambulatory", "outpatient", "virtual", "home")
REPORTED_EGFR_CODES = ("33914-3", "62238-1", "98979-8", "48642-3", "48643-1")
MAX_LOG_DISAGREEMENT = float(np.log(1.4))
_OBS_COLUMNS = ["DATE", "PATIENT", "ENCOUNTER", "CODE", "VALUE"]
_CHUNK_ROWS = 1_000_000


@dataclass
class IngestReport:
    """Counts describing what the ingestion step kept and removed."""

    source: str = ""
    patients_in_source: int = 0
    creatinine_rows: int = 0
    patient_days_with_creatinine: int = 0
    removed_acute_care: int = 0
    removed_inconsistent: int = 0
    removed_after_kidney_failure: int = 0
    removed_implausible: int = 0
    removed_spikes: int = 0
    patients_in_cohort: int = 0
    visits_in_cohort: int = 0
    exclusions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_day(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()


def _as_dirs(csv_dirs) -> list[Path]:
    if isinstance(csv_dirs, (str, Path)):
        return [Path(csv_dirs)]
    return [Path(d) for d in csv_dirs]


def _read_multi(dirs: list[Path], name: str, **kwargs) -> pd.DataFrame:
    """Read the same CSV from several Synthea batches and stack them."""
    frames = [pd.read_csv(d / name, **kwargs) for d in dirs]
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _read_observations(dirs: list[Path], codes: set[str]) -> pd.DataFrame:
    """Stream observations.csv from each batch and keep only the requested codes."""
    parts = []
    for d in dirs:
        reader = pd.read_csv(d / "observations.csv", usecols=_OBS_COLUMNS, dtype=str, chunksize=_CHUNK_ROWS)
        for chunk in reader:
            parts.append(chunk[chunk["CODE"].isin(codes)])
    if not parts:
        return pd.DataFrame(columns=_OBS_COLUMNS)
    obs = pd.concat(parts, ignore_index=True)
    obs["DATE"] = _to_day(obs["DATE"])
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    return obs.dropna(subset=["DATE", "VALUE"])


def _map_dates(keys: pd.Series, dates: pd.Series) -> pd.Series:
    """Look up a date per key; works across pandas versions even when ``dates`` is empty."""
    lookup = {k: v for k, v in dates.items() if pd.notna(v)}
    return pd.to_datetime(keys.map(lambda k: lookup.get(k, pd.NaT)), errors="coerce")


def _first_date(frame: pd.DataFrame, codes: tuple[str, ...], date_col: str = "START") -> pd.Series:
    sub = frame[frame["CODE"].isin(codes)]
    if sub.empty:
        return pd.Series(dtype="datetime64[ns]")
    return sub.groupby("PATIENT")[date_col].min()


def _remove_spikes(df: pd.DataFrame, max_log_ratio: float = 0.4) -> tuple[pd.DataFrame, int]:
    """Drop isolated single-result spikes in eGFR.

    A result is a spike when it differs from both its previous and next result
    by more than ``max_log_ratio`` on the log scale, in the same direction,
    while those two neighbours agree with each other. A genuine trend changes
    neighbouring results together and is therefore never removed. The first
    and last results of a patient are never flagged because they lack context.
    """
    df = df.sort_values(["patient_id", "date"]).copy()
    log_e = np.log(df["egfr"])
    grp = log_e.groupby(df["patient_id"])
    prev_v, next_v = grp.shift(1), grp.shift(-1)
    up = (log_e - prev_v > max_log_ratio) & (log_e - next_v > max_log_ratio)
    down = (prev_v - log_e > max_log_ratio) & (next_v - log_e > max_log_ratio)
    neighbours_agree = (prev_v - next_v).abs() < max_log_ratio
    spike = (up | down) & neighbours_agree
    spike = spike.fillna(False).astype(bool)
    return df.loc[~spike].copy(), int(spike.sum())


def _asof_attach(visits: pd.DataFrame, measures: pd.DataFrame, name: str, tolerance_days: int) -> pd.DataFrame:
    """Attach the most recent prior measurement within a tolerance window."""
    if measures.empty:
        visits[name] = np.nan
        return visits
    right = (
        measures.rename(columns={"DATE": "date", "PATIENT": "source_id", "VALUE": name})[["source_id", "date", name]]
        .groupby(["source_id", "date"], as_index=False)[name]
        .median()
        .sort_values("date")
    )
    left = visits.sort_values("date")
    merged = pd.merge_asof(
        left,
        right,
        on="date",
        by="source_id",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    return merged


def _finalise(
    visits: pd.DataFrame,
    patients: pd.DataFrame,
    settings: Settings,
    report: IngestReport,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared cohort selection, time axis and patient summary for all sources."""
    ms = settings.model

    plausible = visits["egfr"].between(ms.egfr_floor, ms.egfr_ceiling)
    report.removed_implausible = int((~plausible).sum())
    visits = visits.loc[plausible].copy()

    visits, report.removed_spikes = _remove_spikes(visits)

    visits = visits.sort_values(["patient_id", "date"])
    entry = visits.groupby("patient_id")["date"].transform("min")
    visits["t_years"] = (visits["date"] - entry).dt.days / 365.25

    per = visits.groupby("patient_id").agg(
        n_visits=("egfr", "size"),
        followup_years=("t_years", "max"),
        entry_date=("date", "min"),
        last_date=("date", "max"),
        baseline_egfr=("egfr", "first"),
        last_egfr=("egfr", "last"),
        age_at_entry=("age", "first"),
    )
    reasons = {
        "under_min_age": per["age_at_entry"] < ms.min_age,
        "too_few_visits": per["n_visits"] < ms.min_visits,
        "short_followup": per["followup_years"] < ms.min_followup_years,
    }
    keep = pd.Series(True, index=per.index)
    for label, mask in reasons.items():
        report.exclusions[label] = int((mask & keep).sum())
        keep &= ~mask
    per = per.loc[keep]
    visits = visits[visits["patient_id"].isin(per.index)].copy()

    patients = patients.set_index("patient_id").join(per, how="inner").reset_index()
    first_acr = (
        visits.dropna(subset=["acr"]).groupby("patient_id")["acr"].first()
        if "acr" in visits
        else pd.Series(dtype=float)
    )
    patients["baseline_acr"] = patients["patient_id"].map(first_acr)
    patients["age10"] = (patients["age_at_entry"] - 60.0) / 10.0
    patients["male"] = (patients["sex"] == "M").astype(float)

    kf = patients["kidney_failure_date"]
    patients["kidney_failure_event"] = kf.notna() & (kf > patients["entry_date"])
    patients["years_to_kidney_failure"] = np.where(
        patients["kidney_failure_event"], (kf - patients["entry_date"]).dt.days / 365.25, np.nan
    )

    report.patients_in_cohort = int(len(patients))
    report.visits_in_cohort = int(len(visits))

    visit_cols = ["patient_id", "date", "t_years", "age", "creatinine", "egfr", "acr", "sbp", "hba1c"]
    for col in visit_cols:
        if col not in visits:
            visits[col] = np.nan
    visits = visits[visit_cols].sort_values(["patient_id", "date"]).reset_index(drop=True)
    patients = patients.sort_values("patient_id").reset_index(drop=True)
    return visits, patients


# ---------------------------------------------------------------------------
# Synthea
# ---------------------------------------------------------------------------
def load_synthea(csv_dirs, settings: Settings = SETTINGS) -> tuple[pd.DataFrame, pd.DataFrame, IngestReport]:
    """Build the kidney timeline from one or more Synthea ``output/csv`` directories.

    Several directories can be passed when the population was generated in
    batches (for example with different random seeds); they are combined.
    """
    dirs = _as_dirs(csv_dirs)
    report = IngestReport(source="Synthea CSV export: " + ", ".join(f"{d.parent.name}/{d.name}" for d in dirs))
    L, S = settings.loinc, settings.snomed

    # Patients: only identifier, birth date, death date and sex are read.
    pts = _read_multi(dirs, "patients.csv", usecols=["Id", "BIRTHDATE", "DEATHDATE", "GENDER"], dtype=str)
    pts = pts.drop_duplicates(subset=["Id"])
    pts["BIRTHDATE"] = _to_day(pts["BIRTHDATE"])
    pts["DEATHDATE"] = _to_day(pts["DEATHDATE"])
    report.patients_in_source = int(len(pts))

    wanted = set(L.creatinine) | set(L.acr) | set(L.systolic_bp) | set(L.hba1c) | set(REPORTED_EGFR_CODES)
    obs = _read_observations(dirs, wanted)

    cond = _read_multi(dirs, "conditions.csv", usecols=["START", "PATIENT", "CODE", "DESCRIPTION"], dtype=str)
    cond["START"] = _to_day(cond["START"])
    proc = _read_multi(dirs, "procedures.csv", usecols=["START", "PATIENT", "CODE", "DESCRIPTION"], dtype=str)
    proc["START"] = _to_day(proc["START"])

    # Kidney failure: earliest of ESRD/G5 diagnosis, dialysis or transplant.
    text_match = cond["DESCRIPTION"].str.lower().str.contains("|".join(S.kidney_failure_text), na=False)
    kf_candidates = pd.concat(
        [
            cond.loc[cond["CODE"].isin(S.kidney_failure) | text_match].groupby("PATIENT")["START"].min(),
            _first_date(proc, DIALYSIS_CODES),
            _first_date(proc, TRANSPLANT_CODES),
            _first_date(cond, TRANSPLANT_CODES),
        ],
        axis=1,
    )
    kidney_failure_date = kf_candidates.min(axis=1)

    diabetes_date = _first_date(cond, S.diabetes)
    htn_date = _first_date(cond, S.hypertension)
    ckd_date = _first_date(cond, S.ckd_any)

    # Creatinine: outpatient only, one value per patient-day, serum/plasma preferred.
    cr = obs[obs["CODE"].isin(L.creatinine)].copy()
    report.creatinine_rows = int(len(cr))
    enc = _read_multi(dirs, "encounters.csv", usecols=["Id", "ENCOUNTERCLASS"], dtype=str)
    enc_class = enc.set_index("Id")["ENCOUNTERCLASS"].str.lower()
    cr["encounter_class"] = cr["ENCOUNTER"].map(enc_class)
    outpatient = cr["encounter_class"].isin(OUTPATIENT_CLASSES)
    report.removed_acute_care = int((~outpatient).sum())
    cr = cr.loc[outpatient]
    cr = cr[cr["VALUE"] > 0]

    # Internal consistency between creatinine and a reported eGFR in the same encounter.
    reported = (
        obs[obs["CODE"].isin(REPORTED_EGFR_CODES)]
        .groupby("ENCOUNTER")["VALUE"]
        .median()
        .rename("reported_egfr")
    )
    cr = cr.join(reported, on="ENCOUNTER")
    birth = pts.set_index("Id")["BIRTHDATE"]
    sex0 = pts.set_index("Id")["GENDER"].str.upper().str[0]
    age_tmp = (cr["DATE"] - cr["PATIENT"].map(birth)).dt.days / 365.25
    implied = egfr_ckd_epi_2021(cr["VALUE"].to_numpy(), age_tmp.to_numpy(), cr["PATIENT"].map(sex0).to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        disagreement = np.abs(np.log(implied / cr["reported_egfr"].to_numpy()))
    inconsistent = np.isfinite(disagreement) & (disagreement > MAX_LOG_DISAGREEMENT)
    report.removed_inconsistent = int(inconsistent.sum())
    cr = cr.loc[~inconsistent]
    cr["priority"] = np.where(cr["CODE"] == L.creatinine[0], 0, 1)
    best = cr.groupby(["PATIENT", "DATE"])["priority"].transform("min")
    cr = cr[cr["priority"] == best]
    cr = cr.groupby(["PATIENT", "DATE"], as_index=False)["VALUE"].median()
    report.patient_days_with_creatinine = int(len(cr))

    cr["kf_date"] = _map_dates(cr["PATIENT"], kidney_failure_date)
    after_kf = cr["kf_date"].notna() & (cr["DATE"] >= cr["kf_date"])
    report.removed_after_kidney_failure = int(after_kf.sum())
    cr = cr.loc[~after_kf]

    cr = cr.merge(pts, left_on="PATIENT", right_on="Id", how="inner")
    cr["age"] = (cr["DATE"] - cr["BIRTHDATE"]).dt.days / 365.25
    cr = cr[cr["DEATHDATE"].isna() | (cr["DATE"] <= cr["DEATHDATE"])]
    cr["sex"] = cr["GENDER"].str.upper().str[0]
    cr["egfr"] = egfr_ckd_epi_2021(cr["VALUE"].to_numpy(), cr["age"].to_numpy(), cr["sex"].to_numpy())

    visits = cr.rename(columns={"PATIENT": "source_id", "DATE": "date", "VALUE": "creatinine"})[
        ["source_id", "date", "age", "sex", "creatinine", "egfr"]
    ]
    visits = _asof_attach(visits, obs[obs["CODE"].isin(L.acr)], "acr", tolerance_days=730)
    visits = _asof_attach(visits, obs[obs["CODE"].isin(L.systolic_bp)], "sbp", tolerance_days=365)
    visits = _asof_attach(visits, obs[obs["CODE"].isin(L.hba1c)], "hba1c", tolerance_days=365)

    # Pseudonymous identifiers (stable for a given source file).
    source_ids = sorted(visits["source_id"].unique())
    id_map = {sid: f"P{idx + 1:05d}" for idx, sid in enumerate(source_ids)}
    visits["patient_id"] = visits["source_id"].map(id_map)

    entry_dates = visits.groupby("source_id")["date"].min()
    patients = pd.DataFrame({"source_id": source_ids})
    patients["patient_id"] = patients["source_id"].map(id_map)
    meta = pts.set_index("Id")
    patients["sex"] = patients["source_id"].map(meta["GENDER"]).str.upper().str[0]
    patients["birth_year"] = patients["source_id"].map(meta["BIRTHDATE"]).dt.year
    patients["deceased"] = patients["source_id"].map(meta["DEATHDATE"]).notna()
    entry = _map_dates(patients["source_id"], entry_dates)
    for name, dates in (("diabetes", diabetes_date), ("hypertension", htn_date), ("ckd_diagnosis", ckd_date)):
        d = _map_dates(patients["source_id"], dates)
        patients[name] = (d.notna() & (d <= entry)).astype(float)
    patients["kidney_failure_date"] = _map_dates(patients["source_id"], kidney_failure_date)

    visits, patients = _finalise(visits, patients.drop(columns=["source_id"]), settings, report)
    log.info("Synthea ingestion complete: %s", report.to_dict())
    return visits, patients, report


# ---------------------------------------------------------------------------
# Generic longitudinal table (external cohorts)
# ---------------------------------------------------------------------------
def load_longitudinal_table(
    table: pd.DataFrame,
    patient_col: str = "patient_id",
    date_col: str | None = "date",
    time_years_col: str | None = None,
    egfr_col: str | None = "egfr",
    creatinine_col: str | None = None,
    age_col: str | None = "age",
    sex_col: str | None = "sex",
    acr_col: str | None = None,
    diabetes_col: str | None = None,
    hypertension_col: str | None = None,
    event_time_years_col: str | None = None,
    settings: Settings = SETTINGS,
) -> tuple[pd.DataFrame, pd.DataFrame, IngestReport]:
    """Convert any tidy longitudinal table into the standard timeline format.

    Either ``date_col`` or ``time_years_col`` must identify when each measurement
    was taken. eGFR is used directly when ``egfr_col`` is given; otherwise it is
    computed from ``creatinine_col`` with CKD-EPI 2021 (requires age and sex).
    """
    report = IngestReport(source="Longitudinal table")
    df = table.copy()
    report.patients_in_source = int(df[patient_col].nunique())

    if date_col and date_col in df:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif time_years_col and time_years_col in df:
        origin = pd.Timestamp("2000-01-01")
        df["date"] = origin + pd.to_timedelta(df[time_years_col].astype(float) * 365.25, unit="D")
    else:
        raise ValueError("Provide date_col or time_years_col.")

    df["sex"] = df[sex_col].astype(str).str.upper().str[0] if sex_col and sex_col in df else "U"
    df["age"] = pd.to_numeric(df[age_col], errors="coerce") if age_col and age_col in df else np.nan
    if egfr_col and egfr_col in df:
        df["egfr"] = pd.to_numeric(df[egfr_col], errors="coerce")
        df["creatinine"] = pd.to_numeric(df[creatinine_col], errors="coerce") if creatinine_col else np.nan
    elif creatinine_col and creatinine_col in df:
        df["creatinine"] = pd.to_numeric(df[creatinine_col], errors="coerce")
        df["egfr"] = egfr_ckd_epi_2021(df["creatinine"].to_numpy(), df["age"].to_numpy(), df["sex"].to_numpy())
    else:
        raise ValueError("Provide egfr_col or creatinine_col.")
    df["acr"] = pd.to_numeric(df[acr_col], errors="coerce") if acr_col and acr_col in df else np.nan
    df["sbp"] = np.nan
    df["hba1c"] = np.nan

    df = df.dropna(subset=["date", "egfr"])
    df["patient_id"] = df[patient_col].astype(str)
    report.patient_days_with_creatinine = int(len(df))
    visits = df.groupby(["patient_id", "date"], as_index=False).agg(
        age=("age", "median"), sex=("sex", "first"), creatinine=("creatinine", "median"),
        egfr=("egfr", "median"), acr=("acr", "median"), sbp=("sbp", "median"), hba1c=("hba1c", "median"),
    )

    first = df.sort_values("date").groupby("patient_id").first()
    patients = pd.DataFrame({"patient_id": first.index})
    patients["sex"] = patients["patient_id"].map(first["sex"])
    patients["birth_year"] = np.nan
    patients["deceased"] = False
    patients["diabetes"] = (
        patients["patient_id"].map(pd.to_numeric(first[diabetes_col], errors="coerce")) if diabetes_col else np.nan
    )
    patients["hypertension"] = (
        patients["patient_id"].map(pd.to_numeric(first[hypertension_col], errors="coerce")) if hypertension_col else np.nan
    )
    patients["ckd_diagnosis"] = np.nan
    if event_time_years_col and event_time_years_col in df:
        ev = pd.to_numeric(first[event_time_years_col], errors="coerce")
        entry = first["date"]
        patients["kidney_failure_date"] = patients["patient_id"].map(
            entry + pd.to_timedelta(ev * 365.25, unit="D")
        )
    else:
        patients["kidney_failure_date"] = pd.NaT

    visits, patients = _finalise(visits, patients, settings, report)
    return visits, patients, report


# ---------------------------------------------------------------------------
# Conversion to model inputs
# ---------------------------------------------------------------------------
def to_series(visits: pd.DataFrame, patients: pd.DataFrame, covariate_names) -> list:
    """Convert the timeline tables into a list of PatientSeries objects."""
    from .twin import PatientSeries

    cov = patients.set_index("patient_id")
    out = []
    for pid, sub in visits.groupby("patient_id", sort=True):
        row = cov.loc[pid] if pid in cov.index else None
        covariates = {}
        for name in covariate_names:
            value = row[name] if row is not None and name in row else 0.0
            covariates[name] = 0.0 if pd.isna(value) else float(value)
        out.append(PatientSeries(pid, sub["t_years"].to_numpy(), sub["egfr"].to_numpy(), covariates))
    return out
