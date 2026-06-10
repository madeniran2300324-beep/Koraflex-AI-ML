"""Day 7 — Fast rule-based layer (runs before/with the ML model)."""
from __future__ import annotations

from app.schemas.common import TransactionPayload


def evaluate_rules(tx: TransactionPayload, features: dict) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    triggered: list[str] = []

    if features.get("tx_count_5m", 0) >= 4:
        score += 35
        reasons.append("4+ transactions in 5 minutes")
        triggered.append("velocity_5m")

    if features.get("high_value_new_user"):
        score += 40
        reasons.append("High-value purchase by <7d old account")
        triggered.append("new_user_high_value")

    if features.get("device_tx_count_1h", 0) >= 6:
        score += 25
        reasons.append("Device making 6+ tx/hour")
        triggered.append("device_velocity")

    if features.get("ip_tx_count_1h", 0) >= 8:
        score += 20
        reasons.append("IP making 8+ tx/hour")
        triggered.append("ip_velocity")

    if tx.amount <= 0:
        score += 100
        reasons.append("Non-positive amount")
        triggered.append("invalid_amount")

    # New account under 7 days making any significant purchase
    if features.get("is_new_user") and tx.amount > 50_000:
        score += 20
        reasons.append("Account under 7 days old making significant purchase")
        triggered.append("new_user_elevated")

    # Device shared across multiple accounts — strong fraud signal
    if features.get("device_unique_users_24h", 1) >= 3:
        score += 30
        reasons.append(f"Device linked to {int(features['device_unique_users_24h'])} accounts today")
        triggered.append("device_sharing")

    # Account less than 1 hour old attempting any transaction
    if tx.account_age_minutes < 60:
        score += 35
        reasons.append("Account less than 1 hour old")
        triggered.append("fresh_account")

    return min(100, score), reasons, triggered