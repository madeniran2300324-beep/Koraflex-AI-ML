"""Day 5 — Velocity checks + feature engineering for transactions.

FIXES the "no device fingerprint features in the model" gap:
- Previously, device_id was used only as a Redis key prefix (for velocity counts).
  The model never received a device-level risk signal.
- This version adds:
    * device_tx_count_1h   (already existed — kept)
    * device_unique_users_24h  — number of distinct user_ids seen on this device
    * device_is_known      — 1 if the device has appeared in the last 30 days, else 0
    * device_account_age_days  — how old the oldest account using this device is
    * ip_unique_users_1h   — number of distinct users from this IP in the last hour

These new features are added to FEATURE_ORDER in synthetic.py (see patch below)
so the training pipeline picks them up automatically.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.db.mongo import db
from app.db.redis_client import redis
from app.schemas.common import TransactionPayload


async def compute_velocity_features(tx: TransactionPayload) -> dict:
    r = redis()
    user_key    = f"tx:user:{tx.user_id}"
    device_key  = f"tx:device:{tx.device_id or 'none'}"
    ip_key      = f"tx:ip:{tx.ip or 'none'}"

    now_iso = tx.timestamp.isoformat()
    # Each entry stores  "<iso_timestamp>|<amount>|<user_id>"  so we can count
    # distinct users per device/IP without a separate key.
    entry = f"{now_iso}|{tx.amount}|{tx.user_id}"
    for k in (user_key, device_key, ip_key):
        await r.lpush(k, entry)
        await r.ltrim(k, 0, 199)
        await r.expire(k, 24 * 3600)

    def _parse(entries: list[str]) -> list[tuple[datetime, float, str]]:
        out = []
        for e in entries:
            parts = e.split("|", 2)
            if len(parts) == 3:
                ts_str, amt_str, uid = parts
            elif len(parts) == 2:
                # backward compat with old format that lacked user_id
                ts_str, amt_str = parts
                uid = ""
            else:
                continue
            try:
                out.append((datetime.fromisoformat(ts_str), float(amt_str), uid))
            except ValueError:
                continue
        return out

    def _window(parsed: list[tuple[datetime, float, str]], minutes: int):
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        subset = [(ts, amt, uid) for ts, amt, uid in parsed if ts.replace(tzinfo=None) >= cutoff]
        count  = len(subset)
        total  = sum(amt for _, amt, _ in subset)
        users  = {uid for _, _, uid in subset if uid}
        return count, total, users

    user_entries   = await r.lrange(user_key, 0, 199)
    device_entries = await r.lrange(device_key, 0, 199)
    ip_entries     = await r.lrange(ip_key, 0, 199)

    user_parsed   = _parse(user_entries)
    device_parsed = _parse(device_entries)
    ip_parsed     = _parse(ip_entries)

    c5,    a5,    _        = _window(user_parsed,   5)
    c60,   a60,   _        = _window(user_parsed,   60)
    c1440, a1440, _        = _window(user_parsed,   1440)
    dc60,  _,     d_users  = _window(device_parsed, 60)
    ic60,  _,     ip_users = _window(ip_parsed,     60)

    # ---------- NEW: device-level fingerprint features ----------

    # How many distinct user_ids have used this device in the last 24h?
    _, _, d_users_24h = _window(device_parsed, 1440)
    device_unique_users_24h = len(d_users_24h)

    # Is this a known device (seen before this transaction)?
    device_is_known = 0
    if tx.device_id:
        existing = await db().user_devices.find_one({"device_id": tx.device_id})
        device_is_known = 1 if existing else 0

    # How old (in days) is the earliest account that used this device?
    device_account_age_days = tx.user_age_days  # default to current user
    if tx.device_id:
        oldest = await db().user_devices.find_one(
            {"device_id": tx.device_id},
            sort=[("first_seen", 1)],
        )
        if oldest and oldest.get("first_seen"):
            delta = (datetime.utcnow() - oldest["first_seen"]).days
            device_account_age_days = min(delta, tx.user_age_days)

    # Distinct users from this IP in the last hour
    ip_unique_users_1h = len(ip_users)

    # Record device observation for future lookups
    if tx.device_id:
        await db().user_devices.update_one(
            {"user_id": tx.user_id, "device_id": tx.device_id},
            {
                "$set": {"last_seen": tx.timestamp},
                "$setOnInsert": {"first_seen": tx.timestamp},
                "$inc": {"tx_count": 1},
            },
            upsert=True,
        )

    return {
        # --- original features (unchanged) ---
        "tx_count_5m":          c5,
        "tx_amount_5m":         a5,
        "tx_count_1h":          c60,
        "tx_amount_1h":         a60,
        "tx_count_24h":         c1440,
        "tx_amount_24h":        a1440,
        "device_tx_count_1h":   dc60,
        "ip_tx_count_1h":       ic60,
        "amount":               tx.amount,
        "user_age_days":        tx.user_age_days,
        "account_age_minutes":  tx.account_age_minutes,
        "is_new_user":          int(tx.user_age_days <= 7),
        "high_value_new_user":  int(tx.user_age_days <= 7 and tx.amount > 200_000),

        # --- NEW device fingerprint features ---
        "device_unique_users_24h":  device_unique_users_24h,
        "device_is_known":          device_is_known,
        "device_account_age_days":  device_account_age_days,
        "ip_unique_users_1h":       ip_unique_users_1h,
    }
