# Data Quality Engine — KoraFlex

Covers Day 1 (validation, freshness, anomaly), Day 2 (profiling, lineage,
dashboard), Day 10 (monitoring alerts).

## Validation rules
`app/services/data_quality/validation.py` ships rules for `Transaction` and
`User` schemas. Extend by appending `FieldRule(...)` to the appropriate list
or pass a custom list to `validate_dataframe()`.

## Freshness
The Java backend (or Mono webhook handler) `POST`s every event to
`/v1/data-quality/mono/event`. `GET /v1/data-quality/freshness` returns
`fresh|stale|unknown` and Slacks ops if age > `MONO_FRESHNESS_MAX_AGE_MIN`.

## Anomalies
`detect_anomalies()` combines robust z-score (MAD) and IQR fences. Use it
during ingestion on `amount`, `account_balance`, etc.

## Profiling
`profile_dataframe()` returns shape, per-column stats (numeric: mean/std/
quartiles; categorical: top values), missingness, and correlations.

## Lineage
Every transformation hop is recorded in `data_lineage` with input/output
hashes so any model input can be traced back to the original Mono event.

## Dashboard
`streamlit run dashboards/data_quality_dashboard.py` — pass-rate over time,
record counts, latest failing issues.
