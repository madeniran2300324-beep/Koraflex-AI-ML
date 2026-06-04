"""Day 6-7 — Orchestrates rules + ML model + anomaly model into one score.

INCLUDES both Fix 6 (timeout enforcement) and Fix 9 (SHAP reason codes).

Changes vs original scoring.py:
- asyncio.wait_for() enforces 480ms latency budget; falls back to rules-only on timeout.
- For 'review' and 'block' decisions, real SHAP values replace the
  feature_importances_ approximation.
- 'allow' decisions still use the cheap path to keep latency down.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from app.core.config import settings
from app.core.logging import get_logger
from app.db.mongo import db
from app.ml.explainability import explain_prediction, format_reason_codes
from app.ml.registry import ModelRegistry
from app.schemas.common import ScoreResponse, TransactionPayload
from app.services.fraud.rules import evaluate_rules
from app.services.fraud.velocity import compute_velocity_features

log = get_logger(__name__)

_BUDGET_S = getattr(settings, "LATENCY_BUDGET_MS", 480) / 1000.0


def _band(score: int) -> str:
    if score >= settings.FRAUD_AUTO_BLOCK_THRESHOLD:
        return "high"
    if score >= settings.FRAUD_REVIEW_THRESHOLD:
        return "medium"
    return "low"


def _decision(score: int) -> str:
    if score >= settings.FRAUD_AUTO_BLOCK_THRESHOLD:
        return "block"
    if score >= settings.FRAUD_REVIEW_THRESHOLD:
        return "review"
    return "allow"


async def _run_pipeline(tx: TransactionPayload):
    features = await compute_velocity_features(tx)
    rule_score, rule_reasons, rule_triggered = evaluate_rules(tx, features)

    model   = ModelRegistry.get("fraud_xgb")
    anomaly = ModelRegistry.get("anomaly_iforest")

    ml_score      = model.predict_proba(features)  if model   else 0.0
    anomaly_score = anomaly.score(features)         if anomaly else 0.0

    blended = int(round((0.7 * ml_score + 0.3 * anomaly_score) * 100))
    final   = max(rule_score, blended)

    reasons = list(rule_reasons)

    decision = _decision(final)
    if model and decision in ("review", "block"):
        # Use real SHAP values for decisions that a human will review
        shap_pairs = explain_prediction(model, features, k=5)
        reasons += format_reason_codes(shap_pairs)
    elif model:
        # Fast path for 'allow' — cheap proxy is fine, no human reads it
        for feat, contrib in model.top_contributing_features(features, k=3):
            reasons.append(f"Feature '{feat}' contributed {contrib:+.2f} to risk")

    if anomaly_score > 0.7:
        reasons.append(f"Anomaly model score {anomaly_score:.2f}")

    model_version = model.version if model else "rules-only"
    return final, reasons, rule_triggered, model_version, ml_score, anomaly_score, features


async def _rules_only_fallback(tx: TransactionPayload):
    features = await compute_velocity_features(tx)
    score, reasons, triggered = evaluate_rules(tx, features)
    return score, reasons, triggered, features


async def score_transaction(tx: TransactionPayload) -> ScoreResponse:
    start      = time.perf_counter()
    timed_out  = False
    features   = {}

    try:
        final, reasons, rule_triggered, model_version, ml_score, anomaly_score, features = (
            await asyncio.wait_for(_run_pipeline(tx), timeout=_BUDGET_S)
        )
    except asyncio.TimeoutError:
        timed_out = True
        elapsed   = (time.perf_counter() - start) * 1000
        log.warning("scoring_timeout", transaction_id=tx.transaction_id, elapsed_ms=round(elapsed, 1))
        final, reasons, rule_triggered, features = await _rules_only_fallback(tx)
        model_version = "rules-only-timeout"
        ml_score      = 0.0
        anomaly_score = 0.0
        reasons.append("ML model skipped: latency budget exceeded")

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    response = ScoreResponse(
        risk_score=final,
        band=_band(final),
        decision=_decision(final),
        reasons=reasons,
        triggered_rules=rule_triggered,
        model_version=model_version,
        latency_ms=round(elapsed_ms, 2),
    )

    await db().fraud_scores.insert_one({
        "transaction_id":  tx.transaction_id,
        "user_id":         tx.user_id,
        "amount":          tx.amount,
        "features":        features,
        "ml_score":        ml_score,
        "anomaly_score":   anomaly_score,
        "final_score":     final,
        "decision":        response.decision,
        "reasons":         reasons,
        "triggered_rules": rule_triggered,
        "model_version":   model_version,
        "latency_ms":      response.latency_ms,
        "timed_out":       timed_out,
        "created_at":      datetime.now(timezone.utc),
    })

    if response.decision == "review":
        await db().review_queue.update_one(
            {"transaction_id": tx.transaction_id},
            {"$set": {
                "transaction_id": tx.transaction_id,
                "user_id":        tx.user_id,
                "risk_score":     final,
                "reasons":        reasons,
                "status":         "pending",
                "created_at":     datetime.now(timezone.utc),
            }},
            upsert=True,
        )

    return response
