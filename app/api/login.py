from fastapi import APIRouter

from app.schemas.common import LoginPayload
from app.services.fraud.ato import score_login

router = APIRouter()


@router.post("/score")
async def score(payload: LoginPayload) -> dict:
    return await score_login(payload)
