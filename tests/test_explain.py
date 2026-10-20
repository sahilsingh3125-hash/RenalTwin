"""Tests for the explanation layer (no network access needed)."""

from nephrotwin.explain import explain, template_explanation

FACTS = {
    "current_egfr_estimate": 42.0, "slope_per_year": -3.2, "slope_lower": -4.5, "slope_upper": -1.9,
    "kdigo_gfr_category": "G3b", "kdigo_risk_band": "High", "prob_rapid_decline": 0.12,
    "confidence": "Moderate", "n_observations": 7,
}


def test_template_has_both_audiences_and_disclaimer():
    text = template_explanation(FACTS)
    assert "### For the patient" in text and "### For the clinician" in text
    assert "Decision support only" in text
    assert "42" in text and "G3b" in text


def test_explain_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = explain(FACTS, api_key=None)
    assert result.source == "template"


def test_explain_survives_a_bad_key():
    result = explain(FACTS, api_key="invalid-key", timeout=5)
    assert result.source in {"template"} or result.source.startswith("llm:")
