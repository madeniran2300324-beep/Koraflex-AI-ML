# KoraFlex AI/ML Service

Fraud Detection + Data Quality engine for the KoraFlex BNPL platform.

Implements the AI/ML track of the 10-day roadmap:
- **Data Quality (Days 1–2, 10):** validation pipeline, freshness checks, statistical anomaly detection, profiling reports, lineage tracking, quality dashboard.
- **Fraud Detection (Days 3–9):** synthetic identity detection, duplicate PII detection, account takeover, device fingerprinting, behavioral biometrics, velocity checks, feature engineering, supervised ML model (XGBoost), transaction anomaly model, real-time scoring API (<500ms), rule-based layer, response thresholds, review queue, feedback loop, network analysis (fraud rings), explainability (reason codes), bias audit.

## Stack
- Python 3.11+, FastAPI, Uvicorn
- scikit-learn, XGBoost, pandas, numpy, scipy
- MongoDB (motor) — shared with the Java backend
- Redis — rate limits, caching, velocity counters
- Streamlit — data-quality + fraud analytics dashboards
- Pydantic v2, structlog, prometheus-client

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Train a baseline model with synthetic data
python -m app.ml.training.train_fraud_model
# Run the API
uvicorn app.main:app --reload --port 8001
# Run the dashboards
streamlit run dashboards/data_quality_dashboard.py
streamlit run dashboards/fraud_analytics_dashboard.py
```

## Endpoints (high level)
- `POST /v1/identity/score`      — synthetic-identity + duplicate-PII score on register
- `POST /v1/login/score`         — ATO + device-fingerprint score on login
- `POST /v1/transactions/score`  — real-time 0–100 fraud score for a checkout (target <500ms)
- `POST /v1/feedback`            — label confirmed fraud/legit for retraining
- `GET  /v1/review-queue`        — medium-risk transactions for manual review
- `POST /v1/data-quality/validate` — run validation pipeline on a payload/batch
- `GET  /v1/data-quality/freshness` — Mono data freshness status
- `GET  /v1/fraud/network/{user_id}` — graph-based ring detection
- `GET  /healthz`                — liveness

See `docs/MODEL_CARD.md` and `docs/DATA_QUALITY.md`.
