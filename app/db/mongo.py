from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def init_mongo() -> None:
    global _client, _db
    _client = AsyncIOMotorClient(settings.MONGODB_URI)
    _db = _client[settings.MONGODB_DB]
    # Useful indexes
    await _db.users.create_index("email", unique=True, sparse=True)
    await _db.users.create_index("phone", sparse=True)
    await _db.fraud_scores.create_index([("user_id", 1), ("created_at", -1)])
    await _db.fraud_feedback.create_index([("transaction_id", 1)], unique=True)
    await _db.review_queue.create_index([("status", 1), ("risk_score", -1)])
    await _db.data_quality_runs.create_index([("created_at", -1)])
    await _db.mono_events.create_index([("received_at", -1)])


async def close_mongo() -> None:
    global _client
    if _client:
        _client.close()


def db() -> AsyncIOMotorDatabase:
    assert _db is not None, "Mongo not initialized"
    return _db
