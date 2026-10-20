"""Tests for the population model and the individual twin."""

import numpy as np
import pytest

from nephrotwin.twin import DigitalTwin, PatientSeries, PopulationPrior


def simulate_cohort(n=150, seed=0, slope_mean=-1.5, slope_sd=1.0, noise=4.0):
    rng = np.random.default_rng(seed)
    cohort = []
    for i in range(n):
        t = np.sort(rng.uniform(0, 8, rng.integers(5, 14)))
        t[0] = 0.0
        level = rng.normal(70, 15)
        slope = rng.normal(slope_mean, slope_sd)
        y = level + slope * t + rng.normal(0, noise, t.size)
        cohort.append(PatientSeries(f"S{i:03d}", t, y, {"age10": rng.normal(), "diabetes": float(rng.random() < 0.3)}))
    return cohort


@pytest.fixture(scope="module")
def prior():
    return PopulationPrior.fit(simulate_cohort(), ("age10", "diabetes"))


def test_em_recovers_population_parameters(prior):
    assert prior.converged
    assert prior.B[0, 1] == pytest.approx(-1.5, abs=0.35)
    assert np.sqrt(prior.sigma2) == pytest.approx(4.0, abs=0.6)
    assert np.sqrt(prior.Sigma[1, 1]) == pytest.approx(1.0, abs=0.35)


def test_kalman_update_equals_batch_bayesian_posterior(prior):
    """With no process noise, sequential updates must equal the closed-form posterior."""
    pr = PopulationPrior.from_dict(prior.to_dict())
    pr.q_level = pr.q_slope = 0.0
    pr.robust_threshold = 0.0
    s = simulate_cohort(1, seed=5)[0]
    twin = DigitalTwin.from_series(pr, s)

    X = np.column_stack([np.ones_like(s.t), s.t])
    mu0 = pr.prior_mean(s.covariates)
    P = np.linalg.inv(np.linalg.inv(pr.Sigma) + X.T @ X / pr.sigma2)
    m = P @ (np.linalg.inv(pr.Sigma) @ mu0 + X.T @ s.egfr / pr.sigma2)
    # The twin's state is [level at t_last, slope]; map the batch result to that time.
    F = np.array([[1.0, s.t[-1]], [0.0, 1.0]])
    np.testing.assert_allclose(twin.mean, F @ m, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(twin.cov, F @ P @ F.T, rtol=1e-6, atol=1e-6)


def test_uncertainty_shrinks_as_results_arrive(prior):
    s = simulate_cohort(1, seed=9)[0]
    sds = [DigitalTwin.from_series(prior, s, k).state_summary()["slope_upper"] - DigitalTwin.from_series(prior, s, k).state_summary()["slope_lower"] for k in range(1, s.n + 1)]
    assert sds[-1] < sds[0]


def test_forecast_interval_widens_with_horizon(prior):
    pr = PopulationPrior.from_dict(prior.to_dict())
    pr.q_level, pr.q_slope = 4.0, 0.5
    twin = DigitalTwin.from_series(pr, simulate_cohort(1, seed=3)[0])
    fc = twin.predict([twin.t_last + 0.5, twin.t_last + 3, twin.t_last + 6])
    widths = (fc["upper"] - fc["lower"]).to_numpy()
    assert np.all(np.diff(widths) > 0)


def test_robust_update_flags_and_limits_an_outlier(prior):
    s = PatientSeries("X", np.arange(6.0), np.array([80, 79, 81, 80, 79, 20.0]))
    robust = DigitalTwin.from_series(prior, s)
    pr = PopulationPrior.from_dict(prior.to_dict())
    pr.robust_threshold = 0.0
    plain = DigitalTwin.from_series(pr, s)
    assert robust.history[-1].down_weighted
    assert robust.current_egfr() > plain.current_egfr()


def test_save_and_load_roundtrip(prior, tmp_path):
    path = tmp_path / "prior.json"
    prior.save(path)
    loaded = PopulationPrior.load(path)
    np.testing.assert_allclose(loaded.B, prior.B)
    np.testing.assert_allclose(loaded.Sigma, prior.Sigma)
    assert loaded.sigma2 == pytest.approx(prior.sigma2)


def test_tuning_and_calibration_run(prior):
    cohort = simulate_cohort(60, seed=11)
    pr = PopulationPrior.from_dict(prior.to_dict())
    result = pr.tune_dynamics(cohort[:40], q_slope_grid=(0.0, 0.5), q_level_grid=(0.0, 4.0), sigma_scale_grid=(1.0,))
    assert "selected_q_slope" in result
    cal = pr.calibrate_intervals(cohort[40:], level=0.8)
    assert 0.3 < cal["interval_scale"] < 3.0


def test_time_to_threshold_is_consistent(prior):
    s = PatientSeries("D", np.arange(8.0), 60 - 6 * np.arange(8.0))  # steep decline
    twin = DigitalTwin.from_series(prior, s)
    out = twin.time_to_threshold(15, horizon_years=10, n_draws=2000)
    assert out["prob_within_horizon"] > 0.5
    assert 0 < out["median_years"] < 10
