from fastapi import APIRouter

from app.services.fraud.network import detect_rings

router = APIRouter()


@router.get("/network/{user_id}")
async def network(user_id: str) -> dict:
    return await detect_rings(user_id)
