# Data card: NephroTwin development cohort

## Source
- **Generator:** Synthea v4.0.0, an open-source synthetic patient generator (Apache License 2.0).
- **Settings:** Massachusetts population model, ages 40 to 85, CSV export only, FHIR export off.
- **Batches:** seed 42 (partial run) and seed 7 (400 requested). Combined by the pipeline.
- **Real patients:** none. All records are synthetic.

## Processing (nephrotwin/ingest.py)
| Step | Rows affected |
|---|---|
| Creatinine results in source | 20,179 |
| Removed: inpatient, emergency or urgent-care encounters | 6,712 |
| Removed: creatinine contradicting the eGFR reported in the same encounter | 7,412 |
| Removed: results after kidney failure (dialysis or transplant) | 1,541 |
| Removed: isolated single-result spikes | 103 |
| Patients excluded: under 18, fewer than 4 results, or under 1 year of follow-up | 46 |

The consistency rule matters for Synthea specifically: some follow-up encounters record a creatinine around 3 mg/dL alongside a laboratory eGFR near 75, which cannot both be true. Without this rule healthy patients would appear to have severe CKD.

## Final cohort
| Characteristic | Value |
|---|---|
| Patients | 404 |
| Laboratory results | 4,274 |
| Median results per patient | 10 |
| Median follow-up | 9.0 years |
| Median age at entry | 54 years |
| Female | 50% |
| Diabetes at entry | 18% |
| Hypertension at entry | 39% |
| CKD G3 or worse at entry | 21% |
| Kidney failure events during follow-up | 28 |

## Privacy
Only a pseudonymous ID, sex, birth year, dates and laboratory values are kept. Name, address and identity-number fields in the source files are never read.

## Known biases
Synthea models a US (Massachusetts) population. Disease prevalence, kidney trajectories and care patterns differ from India. Kidney function is mostly stable in this cohort, which makes simple baselines strong.
