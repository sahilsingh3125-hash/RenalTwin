# Model card: NephroTwin

## Model details
- **Type:** Bayesian state-space model (local linear trend) with a hierarchical population prior fitted by Expectation-Maximisation, robust Kalman updating and split-conformal interval calibration.
- **Inputs:** serial serum creatinine (converted to eGFR with CKD-EPI 2021), age, sex, diabetes and hypertension status at baseline; urine ACR where available (for KDIGO A-category and KFRE).
- **Outputs:** current eGFR estimate, personal slope per year, forecasts with 80% intervals, probability of eGFR below 30 and 15, probability of rapid decline (steeper than 5 per year), KFRE 2- and 5-year risk where valid, what-if scenario projections, plain-language summary.
- **Version:** 1.0.0. Licence: MIT.

## Intended use
- Research, education and demonstration of decision support for monitoring CKD progression in adults with or at risk of CKD.
- Intended users: clinicians and researchers, with the output reviewed by a qualified clinician.

## Out-of-scope use
- Diagnosis, prescribing, dosing, triage or any use on real patients without clinical validation and regulatory clearance.
- Children, pregnancy, acute kidney injury, patients on dialysis or with a kidney transplant.

## Training and evaluation data
- Synthetic adults (Synthea v4.0.0). See [data_card.md](data_card.md).
- Patient-level split: 222 fitting, 61 calibration and 121 test patients. All reported results are on test patients only.

## Performance (held-out test patients)
| Metric | Value |
|---|---|
| Next-result mean absolute error | 6.79 (last value carried forward: 6.67; straight line: 8.93) |
| One-year-plus mean absolute error | 8.80 (last value: 8.57; straight line: 22.15) |
| 80% interval coverage | 82% next result, 84% one year or more ahead |
| Event AUROC, eGFR below 30 at about 2 years | 0.93 (5 events only; preliminary) |

Against a straight line through the patient's own results the twin is significantly more accurate (bootstrap 95% CIs exclude zero). Against carrying the last value forward the difference is not significant.

## Known limitations and risks
- Synthetic data with mostly stable kidney function and few progression events.
- Linear slope within each period; abrupt changes are tracked through process noise, which lags real events.
- Treatment effects are population averages from trials, not individual predictions.
- Not validated in Indian populations; eGFR equations and KFRE calibration may need adjustment.

## Fairness considerations
- CKD-EPI 2021 removes the race coefficient used in older equations.
- Subgroup performance by sex, age and diabetes status has not yet been reported and should be checked before any real-world use.

## Safety measures
- Atypical single results are down-weighted and flagged for repeat testing rather than acted on.
- KFRE is labelled when outside its validated range.
- The language model sees only computed numbers, is told not to invent values or advise doses, and falls back to a fixed template if unavailable.
