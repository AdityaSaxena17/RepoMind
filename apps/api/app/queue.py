from typing import cast

from arq import create_pool
from arq.connections import ArqRedis
from fastapi import Request

from .config import get_settings


async def create_redis_pool() -> ArqRedis:
    settings = get_settings()
    return await create_pool(
        settings.redis_settings,
        default_queue_name=settings.arq_queue_name,
    )


async def get_redis(request: Request) -> ArqRedis:
    redis = getattr(request.app.state, "redis", None)

    if redis is None:
        raise RuntimeError("Redis pool has not been initialized.")

    return cast(ArqRedis, redis)
