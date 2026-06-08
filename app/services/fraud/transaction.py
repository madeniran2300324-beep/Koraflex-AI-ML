"""Transaction fraud scoring service.

Loads the pre-trained XGBoost pipeline from artifacts and scores
each transaction payload against FEATURE_ORDER.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from app.db.mongo import db
from app.db.redis_client import redis
from app.schemas.common import TransactionPayload

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "ml" / "artifacts"

_pipeline = None
_iforest = None


def _load_models():
    global _pipeline, _iforest
    if _pipeline is None:
        _pipeline = joblib.load(ARTIFACT_DIR / "fraud_xgb.joblib")
    if _iforest is None:
        _iforest = joblib.load(ARTIFACT_DIR / "anomaly_iforest.joblib")


FEATURE_ORDER = [
    "amount",
    "user_age_days",
    "account_age_minutes",
    "is_new_user",
    "high_value_new_user",
    "tx_count_5m",
    "tx_amount_5m",
    "tx_count_1h",
    "tx_amount_1h",
    "tx_count_24h",
    "tx_amount_24h",
    "device_tx_count_1h",
    "ip_tx_count_1h",
    "device_unique_users_24h",
    "device_is_known",
    "device_account_age_days",
    "ip_unique_users_1h",
]


async def _get_velocity(r, key: str, window_seconds: int, now: datetime) -> tuple[int, float]:
    """Return (count, total_amount) for transactions in the last window_seconds."""
    raw = await r.lrange(key, 0, 199)
    count = 0
    total = 0.0
    for entry in raw:
        try:
            ts_str, amt_str = entry.split("|", 1)
            ts = datetime.fromisoformat(ts_str)
            delta = (now - ts).total_seconds()
            if delta <= window_seconds:
                count += 1
                total += float(amt_str)
        except Exception:
            continue
    return count, total


async def score_transaction(payload: TransactionPayload) -> dict:
    _load_models()

    r = redis()
    now = payload.timestamp.replace(tzinfo=None) if payload.timestamp.tzinfo else payload.timestamp

    entry = f"{now.isoformat()}|{payload.amount}"

    # --- Velocity windows ---
    tx_key     = f"vel:tx:{payload.user_id}"
    dev_key    = f"vel:dev:{payload.device_id}"
    ip_key     = f"vel:ip:{payload.ip}"

    for key in (tx_key, dev_key, ip_key):
        await r.lpush(key, entry)
        await r.ltrim(key, 0, 199)
        await r.expire(key, 86400)

    tx_count_5m,  tx_amount_5m  = await _get_velocity(r, tx_key,  300,   now)
    tx_count_1h,  tx_amount_1h  = await _get_velocity(r, tx_key,  3600,  now)
    tx_count_24h, tx_amount_24h = await _get_velocity(r, tx_key,  86400, now)
    device_tx_count_1h, _       = await _get_velocity(r, dev_key, 3600,  now)
    ip_tx_count_1h, _           = await _get_velocity(r, ip_key,  3600,  now)

    # --- Device fingerprint signals ---
    dev_users_key = f"dev:users:{payload.device_id}"
    await r.sadd(dev_users_key, payload.user_id)
    await r.expire(dev_users_key, 86400)
    device_unique_users_24h = await r.scard(dev_users_key)

    ip_users_key = f"ip:users:{payload.ip}"
    await r.sadd(ip_users_key, payload.user_id)
    await r.expire(ip_users_key, 3600)
    ip_unique_users_1h = await r.scard(ip_users_key)

    known_devices = await db().user_devices.distinct("device_id", {"user_id": payload.user_id})
    device_is_known = 1 if payload.device_id in known_devices else 0

    dev_doc = await db().user_devices.find_one({"device_id": payload.device_id})
    if dev_doc and dev_doc.get("first_seen"):
        device_account_age_days = max(0, (now - dev_doc["first_seen"].replace(tzinfo=None)).days)
    else:
        device_account_age_days = 0

    # --- Derived flags ---
    is_new_user = 1 if payload.user_age_days <= 7 else 0
    high_value_new_user = 1 if (is_new_user and payload.amount > 200_000) else 0

    features = np.array([[
        payload.amount,
        payload.user_age_days,
        payload.account_age_minutes,
        is_new_user,
        high_value_new_user,
        tx_count_5m,
        tx_amount_5m,
        tx_count_1h,
        tx_amount_1h,
        tx_count_24h,
        tx_amount_24h,
        device_tx_count_1h,
        ip_tx_count_1h,
        device_unique_users_24h,
        device_is_known,
        device_account_age_days,
        ip_unique_users_1h,
    ]], dtype=float)

    proba = float(_pipeline.predict_proba(features)[0, 1])
    anomaly_score = float(_iforest.decision_function(features)[0])

    # Blend: 70% XGB + 30% anomaly (inverted: lower decision_function = more anomalous)
    anomaly_contrib = max(0.0, min(1.0, (0.5 - anomaly_score)))
    blended = min(1.0, 0.70 * proba + 0.30 * anomaly_contrib)
    risk_score = round(blended * 100)

    # --- Band + decision ---
    if risk_score >= 75:
        band, decision = "high", "block"
    elif risk_score >= 45:
        band, decision = "medium", "review"
    else:
        band, decision = "low", "allow"

    # --- Human-readable reasons ---
    reasons: list[str] = []
    triggered_rules: list[str] = []

    if is_new_user:
        reasons.append("New user account (≤7 days old)")
        triggered_rules.append("new_user")
    if high_value_new_user:
        reasons.append("High-value transaction from new user")
        triggered_rules.append("high_value_new_user")
    if tx_count_5m >= 3:
        reasons.append(f"High velocity: {tx_count_5m} transactions in 5 minutes")
        triggered_rules.append("tx_velocity_5m")
    if tx_count_1h >= 8:
        reasons.append(f"High velocity: {tx_count_1h} transactions in 1 hour")
        triggered_rules.append("tx_velocity_1h")
    if device_unique_users_24h >= 3:
        reasons.append(f"Device shared by {device_unique_users_24h} users in 24h")
        triggered_rules.append("device_sharing")
    if ip_unique_users_1h >= 4:
        reasons.append(f"IP shared by {ip_unique_users_1h} users in 1 hour")
        triggered_rules.append("ip_sharing")
    if not device_is_known:
        reasons.append("Unrecognised device")
        triggered_rules.append("unknown_device")
    if proba >= 0.75:
        reasons.append(f"ML model high-confidence fraud signal ({proba:.0%})")
        triggered_rules.append("ml_high_confidence")

    import time
    start = time.time()

    # --- Persist ---
    doc = {
        **payload.model_dump(),
        "risk_score":       risk_score,
        "band":             band,
        "decision":         decision,
        "reasons":          reasons,
        "triggered_rules":  triggered_rules,
        "model_version":    "fraud_xgb_v1",
        "created_at":       datetime.now(timezone.utc),
    }
    latency_ms = round((time.time() - start) * 1000, 2)

    if decision in ("block", "review"):
        await db().review_queue.insert_one({**doc, "status": "pending"})
    await db().fraud_scores.insert_one(doc)

    return {
        "risk_score":       risk_score,
        "band":             band,
        "decision":         decision,
        "reasons":          reasons,
        "triggered_rules":  triggered_rules,
        "model_version":    "fraud_xgb_v1",
        "latency_ms":       latency_ms,
    }
