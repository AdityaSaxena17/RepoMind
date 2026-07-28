import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from arq.connections import RedisSettings


@dataclass(frozen=True)
class Settings:
    redis_host: str
    redis_port: int
    redis_database: int
    arq_queue_name: str
    database_url: str
    internal_api_credential: str | None
    openai_api_key: str | None
    openai_embeddings_model: str
    openai_chat_model: str | None

    @property
    def redis_settings(self) -> RedisSettings:
        return RedisSettings(
            host=self.redis_host,
            port=self.redis_port,
            database=self.redis_database,
        )


def normalize_asyncpg_database_url(database_url: str) -> str:
    """Remove Prisma-only query params before passing DATABASE_URL to asyncpg."""
    parsed = urlsplit(database_url)
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key != "schema"
        ],
    )
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            query,
            parsed.fragment,
        ),
    )


def normalize_openai_chat_model(model: str | None) -> str | None:
    """Convert common OpenAI display labels into API model identifiers."""
    if model is None:
        return None

    normalized = model.strip()
    if not normalized:
        return None

    if normalized.lower() == "gpt-4o mini":
        return "gpt-4o-mini"

    return normalized


@lru_cache
def get_settings() -> Settings:
    database_url = (
        os.getenv("API_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql://repomind:repomind_password@localhost:5433/RepoMind"
    )

    return Settings(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_database=int(os.getenv("REDIS_DATABASE", "0")),
        arq_queue_name=os.getenv("ARQ_QUEUE_NAME", "repomind:indexing"),
        database_url=normalize_asyncpg_database_url(database_url),
        internal_api_credential=os.getenv("INTERNAL_API_CREDENTIAL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_embeddings_model=os.getenv(
            "OPENAI_EMBEDDINGS_MODEL",
            "text-embedding-3-small",
        ),
        openai_chat_model=normalize_openai_chat_model(os.getenv("OPENAI_CHAT_MODEL")),
    )
