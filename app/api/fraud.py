from fastapi import APIRouter, Query

from app.db.mongo import db
from app.schemas.common import ScoreResponse, TransactionPayload
from app.services.fraud.transaction import score_transaction

router = APIRouter()


@router.post("/score", response_model=ScoreResponse)
async def score(payload: TransactionPayload) -> ScoreResponse:
    return await score_transaction(payload)


@router.get("/review-queue")
async def review_queue(limit: int = Query(50, ge=1, le=200)) -> list[dict]:
    cursor = db().review_queue.find({"status": "pending"}).sort("risk_score", -1).limit(limit)
    items = []
    async for d in cursor:
        d["_id"] = str(d["_id"])
        items.append(d)
    return items
