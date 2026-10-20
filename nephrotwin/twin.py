"""Patient-level digital twin of kidney function.

Model
-----
The twin is a Bayesian state-space model. Its hidden state at time t is the
patient's true kidney function and its current rate of change:

    x_t = [level_t, slope_t]

Between two laboratory results separated by dt years the state evolves as

    level_{t+dt} = level_t + slope_t * dt + w_level
    slope_{t+dt} = slope_t + w_slope

where the process noise (w_level, w_slope) lets the rate of decline change over
time, for example when a stable patient begins to progress. Each measured eGFR
is the true level plus measurement noise:

    eGFR_t = level_t + e_t,     e_t ~ N(0, sigma^2)

The starting state of every patient comes from a hierarchical population
prior: [level_0, slope_0] ~ N(B' z, Sigma), where z holds baseline
covariates (age, diabetes, hypertension, sex). B, Sigma and sigma^2 are fitted
by Expectation-Maximisation of the corresponding linear mixed-effects model.
The process-noise variances are then chosen to maximise the one-step-ahead
predictive likelihood on the fitting cohort, and prediction intervals are
calibrated by split-conformal prediction on a separate calibration cohort.

The twin updates its belief with an exact Kalman filter after every new result.
With zero process noise it reduces to the classic random-intercept,
random-slope model; linear eGFR slope is the regulator-accepted surrogate
endpoint for CKD progression (Levey AS et al., Am J Kidney Dis 2020).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import EGFR_KIDNEY_FAILURE, RAPID_DECLINE_PER_YEAR


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class PatientSeries:
    """Longitudinal eGFR observations for one patient."""

    patient_id: str
    t: np.ndarray                      # years since first observation
    egfr: np.ndarray                   # mL/min/1.73m2
    covariates: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.egfr = np.asarray(self.egfr, dtype=float)
        order = np.argsort(self.t, kind="stable")
        self.t, self.egfr = self.t[order], self.egfr[order]

    @property
    def n(self) -> int:
        return int(self.t.size)


def design_matrix(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.column_stack([np.ones_like(t), t])


def covariate_vector(covariates: dict[str, float], names: Sequence[str]) -> np.ndarray:
    """Intercept followed by the named covariates (missing values default to 0)."""
    values = [1.0]
    for name in names:
        v = covariates.get(name, 0.0)
        values.append(0.0 if v is None or not np.isfinite(v) else float(v))
    return np.asarray(values)


def process_noise(dt: float, q_level: float, q_slope: float) -> np.ndarray:
    """Covariance of the state change over dt years (integrated random walk)."""
    if dt <= 0:
        return np.zeros((2, 2))
    return np.array(
        [
            [q_level * dt + q_slope * dt**3 / 3.0, q_slope * dt**2 / 2.0],
            [q_slope * dt**2 / 2.0, q_slope * dt],
        ]
    )


# ---------------------------------------------------------------------------
# Population layer
# ---------------------------------------------------------------------------
@dataclass
class PopulationPrior:
    """Population model shared by all twins.

    Attributes
    ----------
    covariate_names : baseline covariates in z (an intercept is implied).
    B : (q, 2) regression of starting [level, slope] on z.
    Sigma : (2, 2) between-patient covariance of starting [level, slope].
    sigma2 : measurement variance of a single eGFR result.
    q_level, q_slope : process-noise variances per year (0 = fixed trajectory).
    interval_scale : conformal multiplier applied to prediction-interval widths.
    robust_threshold : standardised surprise above which a result is
        down-weighted (Huber-type robust update); 0 disables it.
    """

    covariate_names: tuple[str, ...]
    B: np.ndarray
    Sigma: np.ndarray
    sigma2: float
    q_level: float = 0.0
    q_slope: float = 0.0
    interval_scale: float = 1.0
    robust_threshold: float = 3.0
    ridge: float = 0.0
    independent_effects: bool = False
    n_patients: int = 0
    n_observations: int = 0
    log_likelihood: float = float("nan")
    iterations: int = 0
    converged: bool = False
    tuning: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    def prior_mean(self, covariates: dict[str, float]) -> np.ndarray:
        return covariate_vector(covariates, self.covariate_names) @ self.B

    # ---- EM fit of the mixed-effects layer ----------------------------------
    @classmethod
    def fit(
        cls,
        patients: Iterable[PatientSeries],
        covariate_names: Sequence[str],
        max_iter: int = 200,
        tol: float = 1e-6,
        ridge: float = 0.0,
        independent_effects: bool = False,
    ) -> "PopulationPrior":
        """Fit B, Sigma and sigma^2 by Expectation-Maximisation.

        ridge : penalty shrinking covariate effects (not the intercept) toward
            zero, i.e. toward the population average. Protects against
            over-fitting when the cohort is small.
        independent_effects : if True, the starting level and slope are
            modelled as uncorrelated, which prevents regression-to-the-mean
            artefacts when early results are noisy.
        """
        patients = [p for p in patients if p.n >= 1]
        if len(patients) < 10:
            raise ValueError("At least 10 patients are needed to fit the population prior.")
        names = tuple(covariate_names)

        X_list = [design_matrix(p.t) for p in patients]
        y_list = [p.egfr for p in patients]
        Z = np.vstack([covariate_vector(p.covariates, names) for p in patients])
        n_obs = int(sum(y.size for y in y_list))

        init = []
        for X, y in zip(X_list, y_list):
            if y.size >= 2 and np.ptp(X[:, 1]) > 0:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            else:
                coef = np.array([y.mean(), 0.0])
            init.append(coef)
        init = np.asarray(init)
        B = np.linalg.lstsq(Z, init, rcond=None)[0]
        Sigma = np.cov((init - Z @ B).T) + np.diag([1.0, 0.1])
        if independent_effects:
            Sigma = np.diag(np.diag(Sigma))
        within = [np.var(y) for y in y_list if y.size > 1]
        sigma2 = max(float(np.mean(within)) * 0.5 if within else 25.0, 1.0)

        converged, iteration = False, 0
        for iteration in range(1, max_iter + 1):
            Sigma_inv = np.linalg.inv(Sigma)
            M = np.empty((len(patients), 2))
            V_total = np.zeros((2, 2))
            rss = 0.0
            for i, (X, y) in enumerate(zip(X_list, y_list)):
                V = np.linalg.inv(Sigma_inv + X.T @ X / sigma2)
                m = V @ (Sigma_inv @ (Z[i] @ B) + X.T @ y / sigma2)
                M[i] = m
                V_total += V
                resid = y - X @ m
                rss += float(resid @ resid + np.trace(X @ V @ X.T))

            penalty = ridge * np.diag([0.0] + [1.0] * (Z.shape[1] - 1))
            B_new = np.linalg.solve(Z.T @ Z + penalty, Z.T @ M)
            D = M - Z @ B_new
            Sigma_new = (D.T @ D + V_total) / len(patients)
            Sigma_new = 0.5 * (Sigma_new + Sigma_new.T) + 1e-6 * np.eye(2)
            if independent_effects:
                Sigma_new = np.diag(np.diag(Sigma_new))
            sigma2_new = max(rss / n_obs, 1e-3)

            change = max(
                np.max(np.abs(B_new - B)) / (np.max(np.abs(B)) + 1e-9),
                np.max(np.abs(Sigma_new - Sigma)) / (np.max(np.abs(Sigma)) + 1e-9),
                abs(sigma2_new - sigma2) / sigma2,
            )
            B, Sigma, sigma2 = B_new, Sigma_new, sigma2_new
            if change < tol * 100:
                converged = True
                break

        log_lik = 0.0
        for i, (X, y) in enumerate(zip(X_list, y_list)):
            cov = X @ Sigma @ X.T + sigma2 * np.eye(y.size)
            log_lik += float(stats.multivariate_normal.logpdf(y, mean=X @ (Z[i] @ B), cov=cov, allow_singular=True))
        prev_ll = log_lik

        return cls(
            covariate_names=names,
            B=B,
            Sigma=Sigma,
            sigma2=float(sigma2),
            ridge=float(ridge),
            independent_effects=bool(independent_effects),
            n_patients=len(patients),
            n_observations=n_obs,
            log_likelihood=float(prev_ll),
            iterations=iteration,
            converged=converged,
        )

    # ---- tuning of the dynamic layer ----------------------------------------
    def predictive_log_likelihood(self, patients: Sequence[PatientSeries], long_horizon: float = 1.0) -> dict:
        """Out-of-sample predictive log density at short and long horizons.

        From every origin (after each result), the twin scores (a) the next
        result and (b) the first result at least ``long_horizon`` years later.
        Both matter clinically: the next test and the multi-year outlook.
        """
        short, long_, short_err, long_err = [], [], [], []
        for p in patients:
            twin = DigitalTwin(self, p.covariates, p.patient_id)
            for k in range(p.n):
                if k > 0:
                    twin.assimilate(p.t[k - 1], p.egfr[k - 1])
                pred = twin.predict(p.t[k], include_noise=True, calibrated=False).iloc[0]
                short.append(stats.norm.logpdf(p.egfr[k], pred["mean"], max(pred["sd"], 1e-6)))
                if k > 0:
                    short_err.append(abs(p.egfr[k] - pred["mean"]))
                    far = np.where(p.t >= p.t[k - 1] + long_horizon)[0]
                    if far.size:
                        j = far[0]
                        pj = twin.predict(p.t[j], include_noise=True, calibrated=False).iloc[0]
                        long_.append(stats.norm.logpdf(p.egfr[j], pj["mean"], max(pj["sd"], 1e-6)))
                        long_err.append(abs(p.egfr[j] - pj["mean"]))
        s_mean = float(np.mean(short)) if short else float("nan")
        l_mean = float(np.mean(long_)) if long_ else s_mean
        s_mae = float(np.mean(short_err)) if short_err else float("nan")
        l_mae = float(np.mean(long_err)) if long_err else s_mae
        return {
            "short": s_mean,
            "long": l_mean,
            "combined": 0.5 * (s_mean + l_mean),
            "short_mae": s_mae,
            "long_mae": l_mae,
            "combined_mae": 0.5 * (s_mae + l_mae),
        }

    def tune_dynamics(
        self,
        patients: Sequence[PatientSeries],
        q_slope_grid: Sequence[float] = (0.0, 0.1, 0.5, 2.0),
        q_level_grid: Sequence[float] = (0.0, 1.0, 4.0, 16.0),
        sigma_scale_grid: Sequence[float] = (0.6, 0.8, 1.0),
    ) -> dict:
        """Choose process noise and measurement variance by out-of-sample predictive density.

        The objective weights next-result and one-year-ahead accuracy equally, so
        the twin is not tuned only for short-term tracking.
        """
        base_sigma2 = self.sigma2
        results = []
        for qs in q_slope_grid:
            for ql in q_level_grid:
                for sc in sigma_scale_grid:
                    self.q_slope, self.q_level, self.sigma2 = qs, ql, base_sigma2 * sc
                    results.append((self.predictive_log_likelihood(patients), qs, ql, sc))
        best, qs, ql, sc = max(results, key=lambda r: r[0]["combined"])
        static = next(r[0] for r in results if r[1] == 0.0 and r[2] == 0.0 and r[3] == 1.0)
        self.q_slope, self.q_level, self.sigma2 = qs, ql, base_sigma2 * sc
        self.tuning = {
            "objective": "mean of next-result and one-year-ahead log predictive density",
            "selected_q_slope": qs,
            "selected_q_level": ql,
            "selected_sigma_scale": sc,
            "selected_score": best,
            "static_model_score": static,
            "grid_size": len(results),
        }
        return self.tuning

    def calibrate_intervals(self, patients: Sequence[PatientSeries], level: float = 0.80) -> dict:
        """Split-conformal calibration of the prediction-interval width."""
        scores = []
        for p in patients:
            twin = DigitalTwin(self, p.covariates, p.patient_id)
            for k in range(1, p.n):
                twin.assimilate(p.t[k - 1], p.egfr[k - 1])
                pred = twin.predict(p.t[k], include_noise=True, calibrated=False).iloc[0]
                scores.append(abs(p.egfr[k] - pred["mean"]) / max(pred["sd"], 1e-6))
        scores = np.asarray(scores)
        if scores.size < 20:
            self.interval_scale = 1.0
        else:
            n = scores.size
            q = np.quantile(scores, min(1.0, np.ceil((n + 1) * level) / n))
            self.interval_scale = float(q / stats.norm.ppf(0.5 + level / 2.0))
        self.calibration = {"level": level, "n_scores": int(scores.size), "interval_scale": self.interval_scale}
        return self.calibration

    # ---- summaries and persistence ------------------------------------------
    def summary(self) -> dict:
        sd = np.sqrt(np.diag(self.Sigma))
        effects = {}
        for j, name in enumerate(("(intercept)",) + self.covariate_names):
            effects[name] = {"starting_egfr": float(self.B[j, 0]), "slope_per_year": float(self.B[j, 1])}
        return {
            "n_patients": self.n_patients,
            "n_observations": self.n_observations,
            "covariate_effects": effects,
            "between_patient_sd_level": float(sd[0]),
            "between_patient_sd_slope": float(sd[1]),
            "level_slope_correlation": float(self.Sigma[0, 1] / (sd[0] * sd[1])),
            "measurement_sd": float(np.sqrt(self.sigma2)),
            "process_noise_sd_slope_per_year": float(np.sqrt(self.q_slope)),
            "process_noise_sd_level_per_year": float(np.sqrt(self.q_level)),
            "conformal_interval_scale": self.interval_scale,
            "em_iterations": self.iterations,
            "em_converged": self.converged,
            "tuning": self.tuning,
            "calibration": self.calibration,
        }

    def to_dict(self) -> dict:
        return {
            "covariate_names": list(self.covariate_names),
            "B": self.B.tolist(),
            "Sigma": self.Sigma.tolist(),
            "sigma2": self.sigma2,
            "q_level": self.q_level,
            "q_slope": self.q_slope,
            "interval_scale": self.interval_scale,
            "robust_threshold": self.robust_threshold,
            "ridge": self.ridge,
            "independent_effects": self.independent_effects,
            "n_patients": self.n_patients,
            "n_observations": self.n_observations,
            "log_likelihood": self.log_likelihood,
            "iterations": self.iterations,
            "converged": self.converged,
            "tuning": self.tuning,
            "calibration": self.calibration,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PopulationPrior":
        return cls(
            covariate_names=tuple(d["covariate_names"]),
            B=np.asarray(d["B"], dtype=float),
            Sigma=np.asarray(d["Sigma"], dtype=float),
            sigma2=float(d["sigma2"]),
            q_level=float(d.get("q_level", 0.0)),
            q_slope=float(d.get("q_slope", 0.0)),
            interval_scale=float(d.get("interval_scale", 1.0)),
            robust_threshold=float(d.get("robust_threshold", 3.0)),
            ridge=float(d.get("ridge", 0.0)),
            independent_effects=bool(d.get("independent_effects", False)),
            n_patients=int(d.get("n_patients", 0)),
            n_observations=int(d.get("n_observations", 0)),
            log_likelihood=float(d.get("log_likelihood", float("nan"))),
            iterations=int(d.get("iterations", 0)),
            converged=bool(d.get("converged", False)),
            tuning=dict(d.get("tuning", {})),
            calibration=dict(d.get("calibration", {})),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PopulationPrior":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Individual twin
# ---------------------------------------------------------------------------
@dataclass
class TwinState:
    """The twin's belief immediately after assimilating a result."""

    n_observed: int
    t: float
    observed_egfr: float
    mean: np.ndarray      # [level, slope]
    cov: np.ndarray       # 2 x 2
    surprise: float = 0.0          # standardised difference from what the twin expected
    down_weighted: bool = False    # True when the result was treated as atypical

    @property
    def level(self) -> float:
        return float(self.mean[0])

    @property
    def level_sd(self) -> float:
        return float(np.sqrt(self.cov[0, 0]))

    @property
    def slope(self) -> float:
        return float(self.mean[1])

    @property
    def slope_sd(self) -> float:
        return float(np.sqrt(self.cov[1, 1]))


class DigitalTwin:
    """Personal kidney-function twin for a single patient."""

    def __init__(self, prior: PopulationPrior, covariates: dict[str, float] | None = None, patient_id: str = ""):
        self.prior = prior
        self.patient_id = patient_id
        self.covariates = dict(covariates or {})
        self.t_current = 0.0
        self.mean = prior.prior_mean(self.covariates).astype(float)
        self.cov = prior.Sigma.copy()
        self.n_observed = 0
        self.history: list[TwinState] = []

    # ---- state propagation ---------------------------------------------------
    def _propagate(self, mean: np.ndarray, cov: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = process_noise(dt, self.prior.q_level, self.prior.q_slope)
        return F @ mean, F @ cov @ F.T + Q

    def assimilate(self, t: float, egfr: float) -> TwinState | None:
        """Kalman update with one new eGFR result taken at time t (years)."""
        if not np.isfinite(egfr):
            return None
        dt = max(float(t) - self.t_current, 0.0)
        self.mean, self.cov = self._propagate(self.mean, self.cov, dt)
        self.t_current = max(float(t), self.t_current)

        innovation = float(egfr - self.mean[0])
        r_var = self.prior.sigma2
        surprise = innovation / np.sqrt(self.cov[0, 0] + r_var)
        c = self.prior.robust_threshold
        down_weighted = bool(c > 0 and abs(surprise) > c and self.n_observed >= 2)
        if down_weighted:
            # Huber-type robust update: inflate the measurement variance so an
            # atypical single result moves the twin only modestly.
            r_var = r_var * (abs(surprise) / c) ** 2

        s = self.cov[0, 0] + r_var
        gain = self.cov[:, 0] / s
        self.mean = self.mean + gain * innovation
        self.cov = self.cov - np.outer(gain, self.cov[0, :])
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.n_observed += 1
        state = TwinState(
            self.n_observed, self.t_current, float(egfr), self.mean.copy(), self.cov.copy(),
            surprise=float(surprise), down_weighted=down_weighted,
        )
        self.history.append(state)
        return state

    def assimilate_many(self, t: Iterable[float], egfr: Iterable[float]) -> None:
        for ti, yi in zip(t, egfr):
            self.assimilate(ti, yi)

    @classmethod
    def from_series(cls, prior: PopulationPrior, series: PatientSeries, k: int | None = None) -> "DigitalTwin":
        twin = cls(prior, series.covariates, series.patient_id)
        k = series.n if k is None else max(0, min(k, series.n))
        twin.assimilate_many(series.t[:k], series.egfr[:k])
        return twin

    # ---- prediction ------------------------------------------------------------
    @property
    def t_last(self) -> float:
        return self.t_current

    def predict(
        self,
        t: Iterable[float] | float,
        level: float = 0.80,
        include_noise: bool = False,
        calibrated: bool = True,
    ) -> pd.DataFrame:
        """Forecast eGFR at future times (times before the last result are clamped).

        include_noise=False: band for the patient's true kidney function.
        include_noise=True:  band for a single future laboratory result.
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        means, sds = np.empty(t_arr.size), np.empty(t_arr.size)
        for i, ti in enumerate(t_arr):
            m, P = self._propagate(self.mean, self.cov, max(ti - self.t_current, 0.0))
            var = P[0, 0] + (self.prior.sigma2 if include_noise else 0.0)
            means[i], sds[i] = m[0], np.sqrt(max(var, 0.0))
        scale = self.prior.interval_scale if calibrated else 1.0
        z = stats.norm.ppf(0.5 + level / 2.0) * scale
        return pd.DataFrame({"t": t_arr, "mean": means, "lower": means - z * sds, "upper": means + z * sds, "sd": sds * scale})

    def current_egfr(self) -> float:
        return float(self.mean[0])

    def slope_interval(self, level: float = 0.80) -> tuple[float, float, float]:
        z = stats.norm.ppf(0.5 + level / 2.0)
        m, s = self.mean[1], np.sqrt(self.cov[1, 1])
        return float(m), float(m - z * s), float(m + z * s)

    def prob_rapid_decline(self, threshold: float = RAPID_DECLINE_PER_YEAR) -> float:
        """P(current slope is steeper than the KDIGO rapid-progression threshold)."""
        return float(stats.norm.cdf(threshold, loc=self.mean[1], scale=np.sqrt(self.cov[1, 1])))

    def prob_below(self, threshold: float, t: float) -> float:
        """P(true eGFR at time t is below threshold)."""
        row = self.predict(t, calibrated=True).iloc[0]
        return float(stats.norm.cdf(threshold, loc=row["mean"], scale=max(row["sd"], 1e-9)))

    # ---- Monte Carlo trajectories --------------------------------------------
    def sample_paths(
        self,
        years_ahead: float,
        n_draws: int = 4000,
        n_steps: int = 60,
        rng: np.random.Generator | None = None,
        slope_multiplier: np.ndarray | float = 1.0,
        start_offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate future true-eGFR paths from the current state.

        ``slope_multiplier`` (per draw) scales negative slopes from
        ``start_offset`` years onward, which is how treatment scenarios act.
        """
        rng = rng or np.random.default_rng(0)
        grid = np.linspace(0.0, years_ahead, n_steps + 1)
        dt = grid[1] - grid[0]
        draws = rng.multivariate_normal(self.mean, self.cov, size=n_draws)
        level, slope = draws[:, 0].copy(), draws[:, 1].copy()
        mult = np.broadcast_to(np.asarray(slope_multiplier, dtype=float), (n_draws,))
        paths = np.empty((n_draws, grid.size))
        paths[:, 0] = level
        for i in range(1, grid.size):
            active = grid[i - 1] >= start_offset
            eff = np.where(active & (slope < 0), slope * mult, slope)
            level = level + eff * dt + rng.normal(0.0, np.sqrt(self.prior.q_level * dt), n_draws)
            slope = slope + rng.normal(0.0, np.sqrt(self.prior.q_slope * dt), n_draws)
            paths[:, i] = level
        return self.t_current + grid, paths

    def time_to_threshold(
        self,
        threshold: float = EGFR_KIDNEY_FAILURE,
        horizon_years: float = 10.0,
        n_draws: int = 4000,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Distribution of years until true eGFR first falls below threshold."""
        grid, paths = self.sample_paths(horizon_years, n_draws, n_steps=int(horizon_years * 12), rng=rng)
        below = paths < threshold
        crossed = below.any(axis=1)
        first = np.where(crossed, below.argmax(axis=1), -1)
        years = np.where(crossed, grid[np.maximum(first, 0)] - self.t_current, np.inf)
        finite = years[np.isfinite(years)]
        return {
            "threshold": threshold,
            "horizon_years": horizon_years,
            "prob_within_horizon": float(crossed.mean()),
            "median_years": float(np.median(years)),
            "p10_years": float(np.percentile(finite, 10)) if finite.size else float("inf"),
            "p90_years": float(np.percentile(finite, 90)) if finite.size else float("inf"),
        }

    # ---- reporting -------------------------------------------------------------
    def uncertainty_ratio(self) -> float:
        return float(np.sqrt(self.cov[1, 1]) / np.sqrt(self.prior.Sigma[1, 1]))

    def confidence_label(self) -> str:
        ratio = self.uncertainty_ratio()
        if ratio > 0.8:
            return "Low: mostly population-based, more results needed"
        if ratio > 0.5:
            return "Moderate"
        return "High: well personalised to this patient"

    def state_summary(self, level: float = 0.80) -> dict:
        slope, lo, hi = self.slope_interval(level)
        return {
            "patient_id": self.patient_id,
            "n_observations": self.n_observed,
            "years_followed": self.t_current,
            "current_egfr_estimate": self.current_egfr(),
            "current_egfr_sd": float(np.sqrt(self.cov[0, 0])),
            "slope_per_year": slope,
            "slope_lower": lo,
            "slope_upper": hi,
            "prob_rapid_decline": self.prob_rapid_decline(),
            "uncertainty_ratio": self.uncertainty_ratio(),
            "confidence": self.confidence_label(),
        }
