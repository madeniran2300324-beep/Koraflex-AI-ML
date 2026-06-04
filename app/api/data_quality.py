from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.db.mongo import db
from app.services.data_quality.anomaly import detect_anomalies
from app.services.data_quality.freshness import freshness_status, record_mono_event
from app.services.data_quality.profiling import profile_dataframe
from app.services.data_quality.validation import (
    TRANSACTION_RULES,
    USER_RULES,
    validate_dataframe,
    validate_record,
)

router = APIRouter()


@router.post("/validate")
async def validate(payload: dict) -> dict:
    """Body: {"schema":"transaction"|"user", "records":[...]} or single dict."""
    schema = payload.get("schema", "transaction")
    rules = TRANSACTION_RULES if schema == "transaction" else USER_RULES
    records = payload.get("records")
    if records is None:
        raise HTTPException(400, "records required")
    if isinstance(records, dict):
        report = validate_record(records, rules)
    else:
        report = validate_dataframe(pd.DataFrame(records), rules)
    doc = {
        "schema": schema,
        "report": report.to_dict(),
        "created_at": datetime.now(timezone.utc),
    }
    await db().data_quality_runs.insert_one(doc)
    return report.to_dict()


@router.post("/profile")
async def profile(payload: dict) -> dict:
    records = payload.get("records") or []
    if not records:
        raise HTTPException(400, "records required")
    return profile_dataframe(pd.DataFrame(records))


@router.post("/anomalies")
async def anomalies(payload: dict) -> dict:
    values = payload.get("values") or []
    z = float(payload.get("z_threshold", 3.5))
    df = detect_anomalies(pd.Series(values), z_thresh=z)
    return {"n_anomalies": int(len(df)), "rows": df.reset_index().to_dict(orient="records")}


@router.post("/mono/event")
async def mono_event(payload: dict) -> dict:
    await record_mono_event(payload.get("event_type", "unknown"), payload)
    return {"status": "recorded"}


@router.get("/freshness")
async def freshness() -> dict:
    return await freshness_status()
