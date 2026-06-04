from fastapi import APIRouter

from app.ml.registry import ModelRegistry

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "models_loaded": list(ModelRegistry._models.keys()),  # noqa: SLF001
    }
