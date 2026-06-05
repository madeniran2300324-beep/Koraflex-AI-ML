"""Admin API — runtime threshold management for ops team.
Authentication uses the same JWT_SECRET Bearer check as the dashboard router.
All changes are audit-logged to MongoDB (threshold_changes collection).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
from app.db.mongo import db
from app.db.redis_client import redis

log = get_logger(__name__)
router = APIRouter()
security = HTTPBearer()

_THRESHOLD_KEY = "fraud:thresholds"


# ---------------------------------------------------------------------------
# In-process cache — updated whenever PUT /thresholds is called
# ---------------------------------------------------------------------------
_live: dict[str, int] = {}


def _verify_ops_token(creds: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if creds.credentials != settings.JWT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid ops token")
    return "ops"


# ---------------------------------------------------------------------------
# Public helpers — used by scoring.py to read live thresholds
# ---------------------------------------------------------------------------

async def load_thresholds_from_redis() -> None:
    """Called at app startup to restore thresholds persisted in Redis."""
    r = redis()
    raw = await r.get(_THRESHOLD_KEY)
    if raw:
        data = json.loads(raw)
        _live.update(data)
        log.info("thresholds_restored_from_redis", thresholds=_live)
    else:
        # Seed from config defaults
        _live["auto_block"] = settings.FRAUD_AUTO_BLOCK_THRESHOLD
        _live["review"]     = settings.FRAUD_REVIEW_THRESHOLD
        log.info("thresholds_seeded_from_config", thresholds=_live)


def get_auto_block_threshold() -> int:
    return _live.get("auto_block", settings.FRAUD_AUTO_BLOCK_THRESHOLD)


def get_review_threshold() -> int:
    return _live.get("review", settings.FRAUD_REVIEW_THRESHOLD)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ThresholdUpdate(BaseModel):
    auto_block: int = Field(..., ge=50, le=100, description="Score ≥ this → block")
    review:     int = Field(..., ge=20, le=90,  description="Score ≥ this → review queue")
    reason:     str = Field(..., min_length=5,  description="Why the change is being made")

    def model_post_init(self, __context) -> None:
        if self.review >= self.auto_block:
            raise ValueError("review threshold must be strictly less than auto_block threshold")


class ThresholdResponse(BaseModel):
    auto_block:   int
    review:       int
    source:       str
    last_updated: str | None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/thresholds", response_model=ThresholdResponse)
async def get_thresholds(_: str = Depends(_verify_ops_token)) -> ThresholdResponse:
    """Return the currently active fraud thresholds."""
    r = redis()
    raw = await r.get(_THRESHOLD_KEY)
    if raw:
        data = json.loads(raw)
        return ThresholdResponse(
            auto_block=data["auto_block"],
            review=data["review"],
            source="redis",
            last_updated=data.get("updated_at"),
        )
    return ThresholdResponse(
        auto_block=settings.FRAUD_AUTO_BLOCK_THRESHOLD,
        review=settings.FRAUD_REVIEW_THRESHOLD,
        source="config",
        last_updated=None,
    )


@router.put("/thresholds", response_model=ThresholdResponse)
async def update_thresholds(
    body: ThresholdUpdate,
    operator: str = Depends(_verify_ops_token),
) -> ThresholdResponse:
    """Update fraud thresholds at runtime. Changes take effect immediately."""
    updated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "auto_block": body.auto_block,
        "review":     body.review,
        "updated_at": updated_at,
        "updated_by": operator,
    }

    r = redis()
    await r.set(_THRESHOLD_KEY, json.dumps(payload))

    # Update in-process cache
    _live["auto_block"] = body.auto_block
    _live["review"]     = body.review

    # Audit log
    await db().threshold_changes.insert_one({
        **payload,
        "reason":     body.reason,
        "previous":   {
            "auto_block": settings.FRAUD_AUTO_BLOCK_THRESHOLD,
            "review":     settings.FRAUD_REVIEW_THRESHOLD,
        },
        "created_at": datetime.now(timezone.utc),
    })

    log.info(
        "thresholds_updated",
        auto_block=body.auto_block,
        review=body.review,
        reason=body.reason,
    )

    return ThresholdResponse(
        auto_block=body.auto_block,
        review=body.review,
        source="redis",
        last_updated=updated_at,
    )


@router.get("/thresholds/history")
async def threshold_history(
    limit: int = 20,
    _: str = Depends(_verify_ops_token),
) -> list[dict]:
    """Return the last N threshold change events for audit purposes."""
    cursor = db().threshold_changes.find().sort("created_at", -1).limit(limit)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results
