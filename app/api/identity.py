from fastapi import APIRouter

from app.core.alerts import send_alert
from app.schemas.common import IdentityPayload
from app.services.fraud.identity import score_identity

router = APIRouter()


@router.post("/score")
async def score(payload: IdentityPayload) -> dict:
    result = await score_identity(payload)
    if result["risk_score"] >= 70:
        await send_alert(
            "High-risk registration",
            f"user_id={payload.user_id} score={result['risk_score']} reasons={result['reasons']}",
            severity="critical",
        )
    return result
