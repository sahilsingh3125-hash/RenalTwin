"""Tests for ingestion and data-quality rules using a small Synthea-shaped fixture."""

import numpy as np
import pandas as pd
import pytest

from nephrotwin.ingest import load_longitudinal_table, load_synthea


def write_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Id": ["a", "b"], "BIRTHDATE": ["1960-01-01", "1955-06-01"], "DEATHDATE": [None, None],
            "GENDER": ["M", "F"], "FIRST": ["Should", "Never"], "SSN": ["999-00-0000", "999-00-0001"],
        }
    ).to_csv(root / "patients.csv", index=False)

    enc, obs = [], []
    def add(pid, i, date, cls, creat, egfr_reported=None):
        eid = f"{pid}-{i}"
        enc.append({"Id": eid, "ENCOUNTERCLASS": cls})
        obs.append({"DATE": date, "PATIENT": pid, "ENCOUNTER": eid, "CODE": "38483-4", "VALUE": creat})
        if egfr_reported is not None:
            obs.append({"DATE": date, "PATIENT": pid, "ENCOUNTER": eid, "CODE": "33914-3", "VALUE": egfr_reported})

    for year in range(2010, 2018):
        add("a", year, f"{year}-03-01T10:00:00Z", "wellness", 1.0)
    add("a", 1, "2012-06-01T10:00:00Z", "inpatient", 4.5)                     # acute care: excluded
    add("a", 2, "2013-06-01T10:00:00Z", "ambulatory", 3.2, egfr_reported=80)  # inconsistent: excluded
    for year in range(2010, 2018):
        add("b", year, f"{year}-05-01T10:00:00Z", "outpatient", 1.0 + 0.4 * (year - 2010))
    add("b", 99, "2017-09-01T10:00:00Z", "outpatient", 9.0)                    # after kidney failure: excluded

    pd.DataFrame(enc).to_csv(root / "encounters.csv", index=False)
    pd.DataFrame(obs).to_csv(root / "observations.csv", index=False)
    pd.DataFrame(
        {"START": ["2015-01-01", "2017-08-01"], "STOP": [None, None], "PATIENT": ["a", "b"], "ENCOUNTER": ["x", "y"],
         "SYSTEM": ["SNOMED-CT"] * 2, "CODE": ["44054006", "46177005"], "DESCRIPTION": ["Diabetes", "End-stage renal disease (disorder)"]}
    ).to_csv(root / "conditions.csv", index=False)
    pd.DataFrame(
        {"START": ["2017-08-15"], "STOP": [None], "PATIENT": ["b"], "ENCOUNTER": ["z"], "SYSTEM": ["SNOMED-CT"],
         "CODE": ["265764009"], "DESCRIPTION": ["Renal dialysis (procedure)"], "BASE_COST": [1], "REASONCODE": [None], "REASONDESCRIPTION": [None]}
    ).to_csv(root / "procedures.csv", index=False)
    return root


@pytest.fixture()
def synthea_dir(tmp_path):
    return write_fixture(tmp_path / "csv")


def test_quality_rules(synthea_dir):
    visits, patients, report = load_synthea(synthea_dir)
    assert report.removed_acute_care == 1
    assert report.removed_inconsistent == 1
    assert report.removed_after_kidney_failure == 1
    assert visits["creatinine"].max() < 9.0
    assert len(patients) == 2


def test_pseudonymisation_and_columns(synthea_dir):
    visits, patients, _ = load_synthea(synthea_dir)
    assert set(patients["patient_id"]) == {"P00001", "P00002"}
    for frame in (visits, patients):
        assert not {"FIRST", "SSN", "source_id", "Id"} & set(frame.columns)


def test_comorbidity_timing_and_events(synthea_dir):
    _, patients, _ = load_synthea(synthea_dir)
    a = patients[patients["sex"] == "M"].iloc[0]
    b = patients[patients["sex"] == "F"].iloc[0]
    assert a["diabetes"] == 0.0            # diagnosed after cohort entry, so not a baseline covariate
    assert bool(b["kidney_failure_event"])
    assert b["years_to_kidney_failure"] == pytest.approx(7.25, abs=0.1)


def test_multiple_batches_are_combined(tmp_path):
    d1 = write_fixture(tmp_path / "one")
    d2 = write_fixture(tmp_path / "two")
    _, patients, _ = load_synthea([d1, d2])
    assert len(patients) == 2               # identical records are de-duplicated


def test_generic_table_loader():
    rows = []
    for pid in range(12):
        for year in range(6):
            rows.append({"pid": pid, "t": float(year), "egfr": 70 - pid - year, "age": 60, "sex": "F"})
    visits, patients, report = load_longitudinal_table(pd.DataFrame(rows), patient_col="pid", date_col=None, time_years_col="t", egfr_col="egfr")
    assert report.patients_in_cohort == 12
    assert np.allclose(visits.groupby("patient_id")["t_years"].max(), 5.0, atol=0.01)


def test_spike_filter_keeps_trends_and_removes_isolated_spikes():
    from nephrotwin.ingest import _remove_spikes

    dates = pd.date_range("2015-01-01", periods=7, freq="365D")
    trend = pd.DataFrame({"patient_id": "T", "date": dates, "egfr": [80, 60, 45, 34, 26, 20, 15.0]})
    spiky = pd.DataFrame({"patient_id": "S", "date": dates, "egfr": [80, 79, 20, 81, 80, 79, 80.0]})
    kept, removed = _remove_spikes(pd.concat([trend, spiky]))
    assert removed == 1
    assert len(kept[kept["patient_id"] == "T"]) == 7
    assert 20.0 not in kept[kept["patient_id"] == "S"]["egfr"].to_numpy()
