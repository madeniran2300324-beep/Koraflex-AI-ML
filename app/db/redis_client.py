from redis.asyncio import Redis

from app.core.config import settings

_redis: Redis | None = None


async def init_redis() -> None:
    global _redis
    _redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    await _redis.ping()


async def close_redis() -> None:
    if _redis:
        await _redis.close()


def redis() -> Redis:
    assert _redis is not None, "Redis not initialized"
    return _redis
