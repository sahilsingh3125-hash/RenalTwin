"""Tests for the what-if simulator."""

import numpy as np

from nephrotwin.interventions import CATALOGUE, combined_reduction, simulate
from nephrotwin.twin import DigitalTwin, PatientSeries, PopulationPrior


def make_twin():
    prior = PopulationPrior(("age10",), np.array([[70.0, -2.0], [0.0, 0.0]]), np.diag([100.0, 1.0]), 16.0)
    s = PatientSeries("A", np.arange(6.0), 70 - 3 * np.arange(6.0))
    return DigitalTwin.from_series(prior, s)


def test_catalogue_values_are_valid():
    for item in CATALOGUE.values():
        assert 0 <= item.low <= item.mode <= item.high < 1
        assert item.references


def test_combined_reduction_formula():
    a, b = np.array([0.4]), np.array([0.2])
    assert combined_reduction([a, b])[0] == 1 - 0.6 * 0.8


def test_no_intervention_gives_identical_arms():
    out = simulate(make_twin(), [], years_ahead=5, n_draws=500)
    np.testing.assert_allclose(out["usual_care"]["median"], out["scenario"]["median"])


def test_treatment_never_worsens_paired_outcome():
    out = simulate(make_twin(), ["sglt2i", "ras_blockade"], years_ahead=5, n_draws=1000)
    assert out["egfr_gain_lower"] >= -1e-9
    assert out["egfr_end_scenario_median"] > out["egfr_end_usual_median"]
    assert out["prob_kidney_failure_scenario"] <= out["prob_kidney_failure_usual"]


def test_delayed_start_has_smaller_benefit():
    twin = make_twin()
    now = simulate(twin, ["sglt2i"], years_ahead=5, start_offset_years=0.0, n_draws=1000)
    later = simulate(twin, ["sglt2i"], years_ahead=5, start_offset_years=2.0, n_draws=1000)
    assert now["egfr_gain_median"] > later["egfr_gain_median"]
