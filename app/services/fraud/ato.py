"""Day 4 — Account Takeover (ATO) + device fingerprinting + behavioral baseline.

CHANGES vs original:
- Calls record_behavior() and score_behavioral_deviation() from the new
  behavioral.py module instead of just dumping raw data into Redis.
- Behavioral deviation now contributes a real risk score (0-40 pts) with
  human-readable reasons.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.db.mongo import db
from app.db.redis_client import redis
from app.schemas.common import LoginPayload
from app.services.fraud.behavioral import record_behavior, score_behavioral_deviation


async def score_login(payload: LoginPayload) -> dict:
    score = 0
    reasons: list[str] = []
    rules:   list[str] = []

    r = redis()

    # --- Login velocity (Redis sliding window) ---
    key = f"login:vel:{payload.user_id}"
    await r.lpush(key, payload.timestamp.isoformat())
    await r.ltrim(key, 0, 49)
    await r.expire(key, 3600)
    recent = await r.lrange(key, 0, 49)
    last_5min = sum(
        1 for ts in recent
        if datetime.fromisoformat(ts) > datetime.utcnow() - timedelta(minutes=5)
    )
    if last_5min >= 5:
        score += 25
        reasons.append(f"{last_5min} logins in last 5 minutes")
        rules.append("login_velocity")

    # --- Device fingerprint history ---
    known_devices = await db().user_devices.distinct("device_id", {"user_id": payload.user_id})
    if known_devices and payload.device_id not in known_devices:
        score += 20
        reasons.append("New device for this user")
        rules.append("new_device")
    await db().user_devices.update_one(
        {"user_id": payload.user_id, "device_id": payload.device_id},
        {
            "$set": {"last_seen": payload.timestamp, "user_agent": payload.user_agent},
            "$inc": {"seen_count": 1},
            "$setOnInsert": {"first_seen": payload.timestamp},
        },
        upsert=True,
    )

    # --- Geographic shift (impossible travel) ---
    last_login = await db().login_events.find_one(
        {"user_id": payload.user_id}, sort=[("timestamp", -1)]
    )
    if last_login and payload.geo_country and last_login.get("geo_country"):
        if payload.geo_country != last_login["geo_country"]:
            delta = (payload.timestamp - last_login["timestamp"]).total_seconds() / 3600.0
            if delta < 2:
                score += 35
                reasons.append(
                    f"Country change {last_login['geo_country']}→{payload.geo_country} in {delta:.1f}h"
                )
                rules.append("impossible_travel")

    # --- IP blocklist ---
    bad_ip = await r.sismember("ip:blocklist", payload.ip)
    if bad_ip:
        score += 30
        reasons.append("IP on blocklist")
        rules.append("bad_ip")

    # --- FIX: Behavioral biometrics — real baseline + deviation scoring ---
    await record_behavior(
        payload.user_id,
        timestamp=payload.timestamp,
        user_agent=payload.user_agent,
        device_id=payload.device_id,
        geo_country=payload.geo_country,
        ip=payload.ip,
    )
    behav_score, behav_reasons = await score_behavioral_deviation(
        payload.user_id,
        current_ua=payload.user_agent,
        current_hour=payload.timestamp.hour,
        current_country=payload.geo_country,
    )
    if behav_score > 0:
        score += behav_score
        reasons.extend(behav_reasons)
        rules.append("behavioral_deviation")

    final = min(100, score)

    await db().login_events.insert_one({
        **payload.model_dump(),
        "timestamp":      payload.timestamp,
        "risk_score":     final,
        "reasons":        reasons,
        "triggered_rules": rules,
        "created_at":     datetime.now(timezone.utc),
    })

    return {"risk_score": final, "reasons": reasons, "triggered_rules": rules}
