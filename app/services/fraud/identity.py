"""Day 3 — Synthetic identity + duplicate-PII detection."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from app.db.mongo import db
from app.schemas.common import IdentityPayload
from app.utils.hashing import normalize_phone, normalize_str, sha256
from app.utils.similarity import similarity_ratio

DISPOSABLE_DOMAINS = {
    "mailinator.com", "tempmail.com", "10minutemail.com", "guerrillamail.com",
    "yopmail.com", "trashmail.com", "discard.email",
}
SUSPICIOUS_NAME_PATTERNS = [
    re.compile(r"^[a-z]{1,2}$", re.I),         # single/double letter
    re.compile(r"^test", re.I),
    re.compile(r"(.)\1{4,}"),                  # repeated chars
    re.compile(r"\d{3,}"),                     # lots of digits in a name
]


async def score_identity(payload: IdentityPayload) -> dict:
    """Return risk score 0-100 + reasons for a new registration."""
    score = 0
    reasons: list[str] = []
    rules: list[str] = []

    email_local, _, email_domain = payload.email.lower().partition("@")
    norm_name = normalize_str(payload.full_name)
    norm_phone = normalize_phone(payload.phone)

    # --- Disposable email ---
    if email_domain in DISPOSABLE_DOMAINS:
        score += 35
        reasons.append(f"Disposable email domain: {email_domain}")
        rules.append("disposable_email")

    # --- Suspicious / synthetic name ---
    for pat in SUSPICIOUS_NAME_PATTERNS:
        if pat.search(norm_name):
            score += 15
            reasons.append(f"Suspicious name pattern: {norm_name!r}")
            rules.append("suspicious_name")
            break

    # --- BVN / NIN required for full trust ---
    if not payload.bvn and not payload.nin:
        score += 10
        reasons.append("No BVN or NIN provided")
        rules.append("no_kyc_id")

    # --- Duplicate PII (exact + fuzzy) ---
    duplicates = await _find_duplicates(payload, norm_name, norm_phone)
    if duplicates["exact_email"]:
        score += 40
        reasons.append("Email already registered")
        rules.append("dup_email")
    if duplicates["exact_phone"]:
        score += 40
        reasons.append("Phone already registered")
        rules.append("dup_phone")
    if duplicates["exact_bvn"]:
        score += 50
        reasons.append("BVN already linked to another account")
        rules.append("dup_bvn")
    if duplicates["fuzzy_name_phone_match"]:
        score += 25
        reasons.append("Fuzzy match: similar name + phone to existing user")
        rules.append("fuzzy_pii")

    # --- Device repeated across many accounts ---
    if payload.device_id:
        device_count = await db().users.count_documents({"device_id": payload.device_id})
        if device_count >= 3:
            score += 20
            reasons.append(f"Device used by {device_count} accounts")
            rules.append("shared_device")

    score = min(100, score)
    await _persist(payload, score, reasons, rules)
    return {"risk_score": score, "reasons": reasons, "triggered_rules": rules}


async def _find_duplicates(p: IdentityPayload, norm_name: str, norm_phone: str) -> dict:
    coll = db().users
    out = {
        "exact_email": bool(await coll.find_one({"email": p.email.lower()})),
        "exact_phone": bool(await coll.find_one({"phone_normalized": norm_phone})),
        "exact_bvn": bool(p.bvn and await coll.find_one({"bvn_hash": sha256(p.bvn)})),
        "fuzzy_name_phone_match": False,
    }
    if norm_phone:
        prefix = norm_phone[:6]
        cursor = coll.find(
            {"phone_normalized": {"$regex": f"^{prefix}"}},
            {"full_name_normalized": 1},
        ).limit(50)
        async for doc in cursor:
            other = doc.get("full_name_normalized") or ""
            if other and similarity_ratio(norm_name, other) >= 0.85:
                out["fuzzy_name_phone_match"] = True
                break
    return out


async def _persist(p: IdentityPayload, score: int, reasons: list[str], rules: list[str]) -> None:
    await db().identity_scores.insert_one({
        "user_id": p.user_id,
        "risk_score": score,
        "reasons": reasons,
        "triggered_rules": rules,
        "created_at": datetime.now(timezone.utc),
    })
