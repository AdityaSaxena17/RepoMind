import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI

from .config import get_settings
from .queue import create_redis_pool
from .routers.chat import router as chat_router
from .routers.repositories import router as repositories_router


@asynccontextmanager
async def application_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the queue/cache client and database pool for this API process."""
    app.state.redis = await create_redis_pool()
    try:
        app.state.database_pool = await asyncpg.create_pool(
            get_settings().database_url,
            min_size=1,
            max_size=10,
            timeout=10,
            command_timeout=60,
        )
        try:
            yield
        finally:
            await asyncio.wait_for(app.state.database_pool.close(), timeout=10)
    finally:
        await app.state.redis.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="RepoMind API",
        version="0.1.0",
        lifespan=application_lifespan,
    )

    app.include_router(repositories_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "openai_chat_model": get_settings().openai_chat_model or "missing",
        }

    return app


app = create_app()
