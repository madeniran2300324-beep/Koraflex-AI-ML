"""Day 1 — Data freshness monitoring for Mono."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.alerts import send_alert
from app.core.config import settings
from app.db.mongo import db


async def record_mono_event(event_type: str, payload: dict) -> None:
    await db().mono_events.insert_one({
        "event_type": event_type,
        "payload": payload,
        "received_at": datetime.now(timezone.utc),
    })


async def freshness_status() -> dict:
    last = await db().mono_events.find_one(sort=[("received_at", -1)])
    now = datetime.now(timezone.utc)
    if not last:
        return {"status": "unknown", "last_event_at": None, "age_minutes": None}
    last_at = last["received_at"]
    if last_at.tzinfo is None:
        last_at = last_at.replace(tzinfo=timezone.utc)
    age_min = (now - last_at).total_seconds() / 60.0
    stale = age_min > settings.MONO_FRESHNESS_MAX_AGE_MIN
    if stale:
        await send_alert(
            "Mono data is stale",
            f"No Mono events for {age_min:.1f} minutes (threshold {settings.MONO_FRESHNESS_MAX_AGE_MIN}m).",
            severity="critical",
        )
    return {
        "status": "stale" if stale else "fresh",
        "last_event_at": last_at.isoformat(),
        "age_minutes": round(age_min, 2),
        "threshold_minutes": settings.MONO_FRESHNESS_MAX_AGE_MIN,
    }
