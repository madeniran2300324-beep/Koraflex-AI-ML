"""Day 2 — Data lineage tracking: Mono → transformations → model input."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.mongo import db


async def record_stage(
    *,
    pipeline_id: str,
    stage: str,
    source: str,
    record_id: str,
    inputs_hash: str,
    outputs_hash: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist one transformation hop so any model input can be traced back."""
    await db().data_lineage.insert_one({
        "pipeline_id": pipeline_id,
        "stage": stage,
        "source": source,
        "record_id": record_id,
        "inputs_hash": inputs_hash,
        "outputs_hash": outputs_hash,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc),
    })


async def trace(record_id: str) -> list[dict]:
    cursor = db().data_lineage.find({"record_id": record_id}).sort("created_at", 1)
    return [doc async for doc in cursor]
