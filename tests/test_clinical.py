"""Tests for the clinical equations."""

import numpy as np
import pytest

from nephrotwin.clinical import (
    KFRE_NON_NORTH_AMERICA,
    KFRE_ORIGINAL,
    albuminuria_category,
    egfr_ckd_epi_2021,
    gfr_category,
    kdigo_risk_band,
    kfre_4var,
)


def test_ckd_epi_2021_reference_male():
    # 50-year-old man, creatinine 1.0 mg/dL -> about 92 mL/min/1.73m2
    assert egfr_ckd_epi_2021(1.0, 50, "M")[0] == pytest.approx(91.7, abs=0.3)


def test_ckd_epi_2021_reference_female_at_kappa():
    # Creatinine equal to kappa (0.7) removes both creatinine terms.
    expected = 142 * 0.9938**60 * 1.012
    assert egfr_ckd_epi_2021(0.7, 60, "F")[0] == pytest.approx(expected, rel=1e-9)


def test_ckd_epi_is_vectorised_and_handles_invalid_input():
    out = egfr_ckd_epi_2021(np.array([1.0, 0.0, np.nan]), np.array([50, 50, 50]), np.array(["M", "F", "M"]))
    assert out.shape == (3,)
    assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isnan(out[2])


def test_egfr_falls_as_creatinine_rises():
    values = egfr_ckd_epi_2021(np.linspace(0.5, 6.0, 20), 55, "F")
    assert np.all(np.diff(values) < 0)


@pytest.mark.parametrize(
    "egfr,expected",
    [(95, "G1"), (90, "G1"), (75, "G2"), (59.9, "G3a"), (45, "G3a"), (44, "G3b"), (29, "G4"), (14, "G5")],
)
def test_gfr_category(egfr, expected):
    assert gfr_category(egfr) == expected


def test_albuminuria_and_risk_band():
    assert albuminuria_category(10) == "A1"
    assert albuminuria_category(100) == "A2"
    assert albuminuria_category(400) == "A3"
    assert kdigo_risk_band("G3a", "A2") == "High"
    assert kdigo_risk_band("G1", "A1") == "Low"
    assert kdigo_risk_band("G4", "A1") == "Very high"


def test_kfre_matches_published_formula():
    age, egfr, acr = 70.0, 30.0, 300.0
    lp = -0.2201 * (age / 10 - 7.036) + 0.2467 * (1 - 0.5642) - 0.5567 * (egfr / 5 - 7.222) + 0.4510 * (np.log(acr) - 5.137)
    r = kfre_4var(age, "M", egfr, acr, KFRE_ORIGINAL)
    assert r.risk_2yr == pytest.approx(1 - 0.9750 ** np.exp(lp))
    assert r.risk_5yr == pytest.approx(1 - 0.9240 ** np.exp(lp))
    assert r.applicable


def test_kfre_properties():
    base = kfre_4var(65, "F", 40, 100, KFRE_NON_NORTH_AMERICA)
    assert 0 < base.risk_2yr < base.risk_5yr < 1
    assert kfre_4var(65, "F", 25, 100).risk_5yr > base.risk_5yr      # lower eGFR, higher risk
    assert kfre_4var(65, "F", 40, 1000).risk_5yr > base.risk_5yr     # more albuminuria, higher risk
    assert not kfre_4var(65, "F", 75, 100).applicable                 # outside validated range
    assert np.isnan(kfre_4var(65, "F", 40, None).risk_5yr)
