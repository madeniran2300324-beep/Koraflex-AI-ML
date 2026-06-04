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

    return min(100, score), reasons, triggered
