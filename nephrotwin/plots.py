"""Static report figures written by the pipeline (PNG, suitable for documents and slides)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .twin import DigitalTwin, PatientSeries, PopulationPrior  # noqa: E402

COLORS = {
    "twin": "#1F6F43",
    "last_value": "#9AA5B1",
    "patient_ols": "#D08C2E",
    "population": "#6C5B7B",
    "patient_mean": "#5B8DB8",
}
LABELS = {
    "twin": "NephroTwin",
    "last_value": "Last value carried forward",
    "patient_mean": "Patient mean so far",
    "patient_ols": "Straight line through own results",
    "population": "Population average only",
}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def learning_curve(curve: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, key in (("twin_mae", "twin"), ("last_value_mae", "last_value"), ("patient_ols_mae", "patient_ols"), ("population_mae", "population")):
        ax.plot(curve["k"], curve[col], marker="o", lw=2 if key == "twin" else 1.3, color=COLORS[key], label=LABELS[key])
    ax.set_xlabel("Number of results the twin has seen (12 = 12 or more)")
    ax.set_ylabel("Error predicting the next result\n(mean absolute error, mL/min/1.73m²)")
    ax.set_title("The twin learns each patient as results arrive (held-out patients)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def method_comparison(metrics: dict, path: Path) -> None:
    keys = ["twin", "last_value", "patient_mean", "patient_ols", "population"]
    nv = [metrics["next_visit"][k]["mae"] for k in keys]
    ly = [metrics["one_year_plus"][k]["mae"] for k in keys] if metrics.get("one_year_plus") else [np.nan] * len(keys)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, nv, 0.4, label="Next result", color=[COLORS[k] for k in keys])
    ax.bar(x + 0.2, ly, 0.4, label="One year or more ahead", color=[COLORS[k] for k in keys], alpha=0.45)
    ax.set_xticks(x, [LABELS[k] for k in keys], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Mean absolute error (mL/min/1.73m²)")
    ax.set_title("Forecast error on held-out patients (lower is better)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def interval_calibration(curve: pd.DataFrame, nominal: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(curve["k"], curve["coverage"] * 100, marker="o", color=COLORS["twin"], label="Observed coverage")
    ax.axhline(nominal * 100, ls="--", color="#555", lw=1, label=f"Target {nominal:.0%}")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of results the twin has seen")
    ax.set_ylabel("Share of real results inside\nthe twin's interval (%)")
    ax.set_title("Are the twin's uncertainty ranges honest?")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def example_twin(prior: PopulationPrior, series: PatientSeries, path: Path, k_list=(2, 5)) -> None:
    """Show how one patient's forecast tightens as the twin sees more results."""
    fig, axes = plt.subplots(1, len(k_list) + 1, figsize=(12, 3.8), sharey=True)
    for ax, k in zip(axes, list(k_list) + [series.n]):
        twin = DigitalTwin.from_series(prior, series, k)
        grid = np.linspace(twin.t_last, twin.t_last + 3, 40)
        fc = twin.predict(grid)
        ax.fill_between(fc["t"], fc["lower"], fc["upper"], color=COLORS["twin"], alpha=0.18, lw=0)
        ax.plot(fc["t"], fc["mean"], color=COLORS["twin"], lw=2)
        ax.scatter(series.t[:k], series.egfr[:k], color="#222", s=18, zorder=3, label="Seen by twin")
        if k < series.n:
            ax.scatter(series.t[k:], series.egfr[k:], facecolors="none", edgecolors="#888", s=18, label="Not yet seen")
        ax.axhline(60, ls=":", color="#999", lw=1)
        ax.axhline(30, ls=":", color="#C0392B", lw=1)
        ax.set_title(f"After {k} result{'s' if k > 1 else ''}", fontsize=10)
        ax.set_xlabel("Years")
        _style(ax)
    axes[0].set_ylabel("eGFR (mL/min/1.73m²)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Patient {series.patient_id}: forecast with 80% range", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
