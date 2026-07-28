from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import get_settings
from .queue import create_redis_pool
from .routers.chat import router as chat_router
from .routers.repositories import router as repositories_router


@asynccontextmanager
async def redis_lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.redis = await create_redis_pool()

    try:
        yield
    finally:
        await app.state.redis.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="RepoMind API",
        version="0.1.0",
        lifespan=redis_lifespan,
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
