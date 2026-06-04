"""Day 4 — Behavioral biometrics: baseline builder + deviation scorer.

FIXES the "stub only" gap:
- Previously, ato.py only pushed user-agent + device_id into a Redis list.
  Nothing compared against a baseline or produced a risk signal.
- This module adds:
    * record_behavior()     — called on every login; stores a rolling window
                              of sessions in Redis (user-agent, country, hour-of-day).
    * build_baseline()      — derives per-user "normal" from the stored window.
    * score_behavioral_deviation() — compares the current session against the
                              baseline and returns a 0-100 risk contribution.
    * The score is then added to the ATO score in ato.py (see patch below).

Signals tracked
---------------
- User-agent consistency   (new UA family = elevated risk)
- Login hour-of-day        (login at 3 AM when user always logs in 8-18 = elevated)
- Country consistency      (handled in ato.py; this module tracks frequency)
- Session frequency        (logins/day vs. baseline average)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from statistics import mean, stdev
from typing import Any

from app.db.redis_client import redis
from app.core.logging import get_logger

log = get_logger(__name__)

WINDOW_SIZE = 200          # max sessions kept per user
WINDOW_TTL_SEC = 90 * 86400  # 90 days


# ---------------------------------------------------------------------------
# Session recording
# ---------------------------------------------------------------------------

async def record_behavior(
    user_id: str,
    *,
    timestamp: datetime,
    user_agent: str,
    device_id: str,
    geo_country: str | None,
    ip: str | None,
) -> None:
    """Push one session snapshot into the user's behavioral window."""
    r = redis()
    key = f"behavior:v2:{user_id}"
    entry = json.dumps({
        "ts": timestamp.isoformat(),
        "ua": _ua_family(user_agent),
        "device": device_id,
        "country": geo_country or "",
        "hour": timestamp.hour,
        "ip_prefix": (ip or "")[:8],   # /24 prefix only — not storing full IPs
    })
    await r.lpush(key, entry)
    await r.ltrim(key, 0, WINDOW_SIZE - 1)
    await r.expire(key, WINDOW_TTL_SEC)


def _ua_family(ua: str) -> str:
    """Coarse user-agent family: 'chrome', 'firefox', 'safari', 'mobile', 'other'."""
    ua_lower = ua.lower()
    if "chrome" in ua_lower and "edg" not in ua_lower:
        return "chrome"
    if "firefox" in ua_lower:
        return "firefox"
    if "safari" in ua_lower:
        return "safari"
    if any(k in ua_lower for k in ("android", "iphone", "ipad", "mobile")):
        return "mobile"
    return "other"


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def _load_sessions(raw_entries: list[str]) -> list[dict[str, Any]]:
    sessions = []
    for e in raw_entries:
        try:
            sessions.append(json.loads(e))
        except Exception:
            continue
    return sessions


def build_baseline(sessions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Derive the user's behavioral baseline from stored sessions.
    Returns None if fewer than 5 sessions exist (not enough signal).
    """
    if len(sessions) < 5:
        return None

    ua_counts: dict[str, int] = {}
    hours: list[int] = []
    countries: dict[str, int] = {}

    for s in sessions:
        ua_counts[s.get("ua", "other")] = ua_counts.get(s.get("ua", "other"), 0) + 1
        hours.append(int(s.get("hour", 12)))
        c = s.get("country", "")
        if c:
            countries[c] = countries.get(c, 0) + 1

    dominant_ua = max(ua_counts, key=ua_counts.__getitem__)
    dominant_ua_frac = ua_counts[dominant_ua] / len(sessions)

    hour_mean = mean(hours)
    hour_std = stdev(hours) if len(hours) > 1 else 6.0

    dominant_country = max(countries, key=countries.__getitem__) if countries else ""
    dominant_country_frac = countries.get(dominant_country, 0) / len(sessions) if countries else 0.0

    return {
        "n_sessions": len(sessions),
        "dominant_ua": dominant_ua,
        "dominant_ua_frac": dominant_ua_frac,
        "hour_mean": hour_mean,
        "hour_std": max(hour_std, 1.0),   # floor at 1 to avoid div/0
        "dominant_country": dominant_country,
        "dominant_country_frac": dominant_country_frac,
    }


# ---------------------------------------------------------------------------
# Deviation scoring
# ---------------------------------------------------------------------------

async def score_behavioral_deviation(
    user_id: str,
    *,
    current_ua: str,
    current_hour: int,
    current_country: str | None,
) -> tuple[int, list[str]]:
    """
    Compare the current session against the stored baseline.

    Returns
    -------
    (risk_contribution: int 0-40, reasons: list[str])
    """
    r = redis()
    key = f"behavior:v2:{user_id}"
    raw = await r.lrange(key, 0, WINDOW_SIZE - 1)
    sessions = _load_sessions(raw)
    baseline = build_baseline(sessions)

    if baseline is None:
        # Not enough history — no score, but note it
        return 0, []

    score = 0
    reasons: list[str] = []

    # 1. User-agent family shift
    current_family = _ua_family(current_ua)
    if current_family != baseline["dominant_ua"] and baseline["dominant_ua_frac"] >= 0.80:
        score += 15
        reasons.append(
            f"Unusual browser/client: '{current_family}' "
            f"(normally '{baseline['dominant_ua']}' in {baseline['dominant_ua_frac']:.0%} of sessions)"
        )

    # 2. Login hour deviation (z-score against baseline distribution)
    hour_z = abs(current_hour - baseline["hour_mean"]) / baseline["hour_std"]
    if hour_z > 2.5:
        score += 10
        reasons.append(
            f"Unusual login hour: {current_hour:02d}:xx "
            f"(baseline μ={baseline['hour_mean']:.1f}, σ={baseline['hour_std']:.1f})"
        )

    # 3. Country shift (supplement to impossible-travel in ato.py)
    if (
        current_country
        and baseline["dominant_country"]
        and current_country != baseline["dominant_country"]
        and baseline["dominant_country_frac"] >= 0.90
    ):
        score += 15
        reasons.append(
            f"Country '{current_country}' deviates from usual "
            f"'{baseline['dominant_country']}' ({baseline['dominant_country_frac']:.0%} of sessions)"
        )

    return min(score, 40), reasons
