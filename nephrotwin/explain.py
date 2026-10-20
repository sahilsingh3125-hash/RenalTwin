"""Plain-language summaries of the twin's state.

Two audiences are served:

* the patient, in short everyday language, and
* the clinician, as a concise structured note.

When a Groq API key is available, a large language model writes the summary
from a fixed set of numeric facts produced by the twin. The model is instructed
never to invent numbers, diagnose, or recommend drug doses. If no key is
configured, or the request fails, a deterministic template produces the same
two sections so that the application always works.

Configuration (environment variables or Streamlit secrets)
-----------------------------------------------------------
GROQ_API_KEY   API key from https://console.groq.com
GROQ_MODEL     optional model identifier (default: openai/gpt-oss-120b)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = ("openai/gpt-oss-20b",)

SYSTEM_PROMPT = """You are a clinical communication assistant inside a kidney-health decision-support tool.
You receive a JSON object of facts computed by a validated statistical model of one person's kidney function.

Write two sections with exactly these headings:
### For the patient
### For the clinician

Rules:
- Use only the numbers present in the facts. Never invent, round to new values, or add numbers.
- Always state uncertainty when a range is provided.
- Do not diagnose, do not prescribe, and do not mention drug doses.
- Treatment scenarios are model projections based on published trial averages, not personal guarantees. Say so when you mention them.
- Patient section: at most 120 words, warm and plain, reading level of a 12-year-old, no jargon except "eGFR" explained once as "a measure of how well the kidneys filter blood".
- Clinician section: at most 120 words, concise, bullet points allowed, include the KDIGO category and risk band when given.
- End the clinician section with: "Decision support only; confirm with clinical assessment and repeat testing."
"""


@dataclass
class Explanation:
    text: str
    source: str        # "llm:<model>" or "template"


def _get_secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return None


def _fmt(value, digits: int = 0, suffix: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "not available"


def template_explanation(facts: dict) -> str:
    """Deterministic two-audience summary built only from the supplied facts."""
    egfr = facts.get("current_egfr_estimate")
    slope = facts.get("slope_per_year")
    lo, hi = facts.get("slope_lower"), facts.get("slope_upper")
    stage = facts.get("kdigo_gfr_category", "Unknown")
    band = facts.get("kdigo_risk_band", "Unknown")
    p_rapid = facts.get("prob_rapid_decline")
    trend_word = "stable"
    if slope is not None and slope < -1:
        trend_word = "slowly falling" if slope > -3 else "falling"
    if slope is not None and slope <= -5:
        trend_word = "falling quickly"

    patient = [
        "### For the patient",
        f"eGFR is a measure of how well the kidneys filter blood. Your current estimate is about {_fmt(egfr)}.",
        f"Over time your eGFR is {trend_word}: about {_fmt(slope, 1)} points per year "
        f"(likely between {_fmt(lo, 1)} and {_fmt(hi, 1)}).",
    ]
    scen = facts.get("scenario")
    if scen:
        patient.append(
            f"In the model, {', '.join(scen.get('interventions', [])) or 'the selected plan'} could leave your eGFR about "
            f"{_fmt(scen.get('egfr_gain_median'), 1)} points higher after {_fmt(scen.get('years_ahead'))} years. "
            "This is an estimate from research averages, not a promise."
        )
    patient.append("Please discuss these results with your doctor before making any change.")

    clinician = [
        "### For the clinician",
        f"- Estimated eGFR {_fmt(egfr)} mL/min/1.73m2; KDIGO {stage}; risk band: {band}.",
        f"- Personal slope {_fmt(slope, 2)} /yr (80% interval {_fmt(lo, 2)} to {_fmt(hi, 2)}); "
        f"P(rapid decline >5/yr) {_fmt(100 * p_rapid if p_rapid is not None else None, 0, '%')}.",
        f"- Twin confidence: {facts.get('confidence', 'not available')} ({facts.get('n_observations', '?')} results).",
    ]
    kfre = facts.get("kfre_5yr")
    if kfre is not None:
        clinician.append(
            f"- KFRE 4-variable: 2-year {_fmt(100 * facts.get('kfre_2yr', 0), 1, '%')}, 5-year {_fmt(100 * kfre, 1, '%')} "
            f"({facts.get('kfre_note', '')})"
        )
    if facts.get("atypical_results"):
        clinician.append(f"- {facts['atypical_results']} atypical result(s) down-weighted; consider repeat testing.")
    if scen:
        clinician.append(
            f"- Scenario ({', '.join(scen.get('interventions', []))}): median eGFR gain {_fmt(scen.get('egfr_gain_median'), 1)} "
            f"at {_fmt(scen.get('years_ahead'))} y (80% range {_fmt(scen.get('egfr_gain_lower'), 1)} to "
            f"{_fmt(scen.get('egfr_gain_upper'), 1)})."
        )
    clinician.append("Decision support only; confirm with clinical assessment and repeat testing.")
    return "\n".join(patient) + "\n\n" + "\n".join(clinician)


def explain(facts: dict, api_key: str | None = None, model: str | None = None, timeout: float = 30.0) -> Explanation:
    """Return a two-audience explanation, using Groq when available."""
    api_key = api_key or _get_secret("GROQ_API_KEY")
    if not api_key:
        return Explanation(template_explanation(facts), "template")

    try:
        from groq import Groq
    except ImportError:
        log.warning("groq package not installed; using template explanation.")
        return Explanation(template_explanation(facts), "template")

    client = Groq(api_key=api_key, timeout=timeout)
    candidates = [model or _get_secret("GROQ_MODEL") or DEFAULT_MODEL, *FALLBACK_MODELS]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(facts, default=float)},
    ]
    for name in dict.fromkeys(candidates):
        try:
            kwargs = {"model": name, "messages": messages, "temperature": 0.2, "max_completion_tokens": 900}
            if name.startswith("openai/gpt-oss"):
                kwargs["reasoning_effort"] = "low"
                kwargs["include_reasoning"] = False
            response = client.chat.completions.create(**kwargs)
            text = (response.choices[0].message.content or "").strip()
            if text:
                return Explanation(text, f"llm:{name}")
        except Exception as exc:  # network errors, model retirement, rate limits
            log.warning("Groq request with model %s failed: %s", name, exc)
    return Explanation(template_explanation(facts), "template")
