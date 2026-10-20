"""NephroTwin: a personalised digital twin for chronic kidney disease.

The package is organised as a set of small, independently testable layers:

    clinical       Validated clinical equations (CKD-EPI 2021, KDIGO staging, KFRE).
    ingest         Conversion of raw EHR-style records into a longitudinal kidney timeline.
    twin           Hierarchical Bayesian model of each patient's kidney trajectory.
    interventions  Evidence-referenced treatment scenarios for what-if simulation.
    evaluation     Rolling-origin validation against clinical baselines.
    explain        Natural-language summaries for patients and clinicians.
    pipeline       End-to-end command-line build of all artefacts.
"""

__version__ = "1.0.0"
