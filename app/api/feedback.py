from fastapi import APIRouter

from app.schemas.common import FeedbackPayload
from app.services.fraud.feedback import record_feedback

router = APIRouter()


@router.post("")
async def feedback(payload: FeedbackPayload) -> dict:
    return await record_feedback(payload)
