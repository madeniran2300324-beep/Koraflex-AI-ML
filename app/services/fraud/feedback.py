"""Day 8 — Feedback loop: label confirmed fraud/legit for retraining."""
from __future__ import annotations

from datetime import datetime, timezone

from app.db.mongo import db
from app.schemas.common import FeedbackPayload


async def record_feedback(fb: FeedbackPayload) -> dict:
    doc = {
        **fb.model_dump(),
        "created_at": datetime.now(timezone.utc),
    }
    await db().fraud_feedback.update_one(
        {"transaction_id": fb.transaction_id},
        {"$set": doc},
        upsert=True,
    )
    # Close the review-queue item if present
    await db().review_queue.update_one(
        {"transaction_id": fb.transaction_id},
        {"$set": {"status": "resolved", "resolution": "fraud" if fb.is_fraud else "legit"}},
    )
    return {"status": "recorded"}


async def export_training_set() -> list[dict]:
    """Join scored transactions with labels for the next retrain."""
    pipeline = [
        {"$lookup": {
            "from": "fraud_scores",
            "localField": "transaction_id",
            "foreignField": "transaction_id",
            "as": "score",
        }},
        {"$unwind": "$score"},
        {"$project": {
            "_id": 0,
            "transaction_id": 1,
            "is_fraud": 1,
            "features": "$score.features",
            "final_score": "$score.final_score",
        }},
    ]
    return [doc async for doc in db().fraud_feedback.aggregate(pipeline)]
