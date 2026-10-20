"""Treatment scenarios for what-if simulation.

Each scenario changes the patient's personal eGFR slope from a chosen start time.
Effects are expressed as a *relative reduction in the rate of eGFR decline*,
which is how kidney trials report chronic eGFR slope differences. Every effect
is a distribution (triangular: low, most likely, high), not a single number, so
uncertainty about the treatment adds to uncertainty about the patient.

The values below are scenario assumptions derived from published trial-level
slope differences. They are intended for exploring trajectories, not for
prescribing. Treatment effects in an individual can differ substantially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .twin import DigitalTwin


@dataclass(frozen=True)
class Intervention:
    key: str
    name: str
    low: float          # relative slope reduction, lower plausible value
    mode: float         # most likely value
    high: float         # upper plausible value
    evidence: str
    references: str
    applies_when: str

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.high <= self.low:
            return np.full(n, self.mode)
        return rng.triangular(self.low, self.mode, self.high, size=n)


CATALOGUE: dict[str, Intervention] = {
    "sglt2i": Intervention(
        key="sglt2i",
        name="SGLT2 inhibitor",
        low=0.30,
        mode=0.40,
        high=0.55,
        evidence="High (multiple large randomised trials)",
        references=(
            "Heerspink HJL et al. DAPA-CKD eGFR slope analysis, Lancet Diabetes Endocrinol 2021; "
            "EMPA-KIDNEY Collaborative Group, N Engl J Med 2023; "
            "DAPA advKD trial (CKD G4-G5): -2.24 vs -3.67 mL/min/1.73m2 per year."
        ),
        applies_when="CKD with or without type 2 diabetes; benefit larger with albuminuria.",
    ),
    "ras_blockade": Intervention(
        key="ras_blockade",
        name="ACE inhibitor / ARB optimisation",
        low=0.05,
        mode=0.15,
        high=0.25,
        evidence="Moderate (strongest in albuminuric CKD)",
        references="RENAAL (Brenner BM et al., N Engl J Med 2001); IDNT (Lewis EJ et al., N Engl J Med 2001).",
        applies_when="Albuminuria and/or hypertension; requires potassium and creatinine monitoring.",
    ),
    "glp1ra": Intervention(
        key="glp1ra",
        name="GLP-1 receptor agonist (semaglutide)",
        low=0.20,
        mode=0.30,
        high=0.40,
        evidence="High in type 2 diabetes with CKD (single large outcome trial)",
        references="FLOW trial (Perkovic V et al., N Engl J Med 2024).",
        applies_when="Type 2 diabetes with CKD.",
    ),
    "bp_control": Intervention(
        key="bp_control",
        name="Blood pressure to guideline target",
        low=0.00,
        mode=0.10,
        high=0.20,
        evidence="Low to moderate for eGFR slope (benefit mainly in proteinuric CKD)",
        references="KDIGO 2021 Blood Pressure in CKD guideline; MDRD and AASK long-term follow-up.",
        applies_when="Uncontrolled hypertension.",
    ),
}


def combined_reduction(samples: Sequence[np.ndarray]) -> np.ndarray:
    """Combine independent relative reductions multiplicatively.

    Remaining decline = product of (1 - r_k). This assumes independent mechanisms
    and is an explicit modelling assumption, not trial evidence of additivity.
    """
    remaining = np.ones_like(samples[0])
    for s in samples:
        remaining = remaining * (1.0 - s)
    return 1.0 - remaining


def simulate(
    twin: DigitalTwin,
    intervention_keys: Sequence[str],
    years_ahead: float = 5.0,
    start_offset_years: float = 0.0,
    n_draws: int = 4000,
    level: float = 0.80,
    seed: int = 7,
    kidney_failure_threshold: float = 15.0,
) -> dict:
    """Simulate future true eGFR with and without the chosen interventions.

    Both arms use the same random draws (common random numbers), so the
    difference between them reflects the intervention rather than sampling noise.
    """
    n_steps = max(int(years_ahead * 12), 12)
    rng_effect = np.random.default_rng(seed)
    if intervention_keys:
        reductions = combined_reduction([CATALOGUE[k].sample(n_draws, rng_effect) for k in intervention_keys])
    else:
        reductions = np.zeros(n_draws)

    grid, usual = twin.sample_paths(years_ahead, n_draws, n_steps, rng=np.random.default_rng(seed + 1))
    _, scenario = twin.sample_paths(
        years_ahead,
        n_draws,
        n_steps,
        rng=np.random.default_rng(seed + 1),
        slope_multiplier=1.0 - reductions,
        start_offset=start_offset_years,
    )

    lo_q, hi_q = 50 * (1 - level), 100 - 50 * (1 - level)

    def band(paths: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "t": grid,
                "median": np.median(paths, axis=0),
                "lower": np.percentile(paths, lo_q, axis=0),
                "upper": np.percentile(paths, hi_q, axis=0),
            }
        )

    def years_to_failure(paths: np.ndarray) -> np.ndarray:
        below = paths < kidney_failure_threshold
        crossed = below.any(axis=1)
        idx = below.argmax(axis=1)
        return np.where(crossed, grid[idx] - grid[0], np.inf)

    gain = scenario[:, -1] - usual[:, -1]
    t_usual, t_scen = years_to_failure(usual), years_to_failure(scenario)
    delay = np.where(np.isfinite(t_usual), np.minimum(t_scen, years_ahead) - t_usual, 0.0)

    return {
        "usual_care": band(usual),
        "scenario": band(scenario),
        "egfr_end_usual_median": float(np.median(usual[:, -1])),
        "egfr_end_scenario_median": float(np.median(scenario[:, -1])),
        "egfr_gain_median": float(np.median(gain)),
        "egfr_gain_lower": float(np.percentile(gain, lo_q)),
        "egfr_gain_upper": float(np.percentile(gain, hi_q)),
        "prob_kidney_failure_usual": float(np.mean(np.isfinite(t_usual))),
        "prob_kidney_failure_scenario": float(np.mean(np.isfinite(t_scen))),
        "median_delay_years_if_progressing": float(np.median(delay[np.isfinite(t_usual)])) if np.isfinite(t_usual).any() else 0.0,
        "median_relative_reduction": float(np.median(reductions)),
        "interventions": [CATALOGUE[k].name for k in intervention_keys],
        "years_ahead": years_ahead,
    }
