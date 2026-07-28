import os
from dataclasses import dataclass
from functools import lru_cache

from arq.connections import RedisSettings


@dataclass(frozen=True)
class Settings:
    redis_host: str
    redis_port: int
    redis_database: int
    arq_queue_name: str
    database_url: str
    internal_api_credential: str | None
    github_app_id: str | None
    github_app_private_key: str | None
    github_api_base_url: str
    github_api_version: str
    openai_api_key: str | None
    openai_embeddings_model: str

    @property
    def redis_settings(self) -> RedisSettings:
        return RedisSettings(
            host=self.redis_host,
            port=self.redis_port,
            database=self.redis_database,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_database=int(os.getenv("REDIS_DATABASE", "0")),
        arq_queue_name=os.getenv("ARQ_QUEUE_NAME", "repomind:indexing"),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql://repomind:repomind_password@localhost:5433/RepoMind",
        ),
        internal_api_credential=os.getenv("INTERNAL_API_CREDENTIAL"),
        github_app_id=os.getenv("GITHUB_APP_ID"),
        github_app_private_key=os.getenv("GITHUB_APP_PRIVATE_KEY"),
        github_api_base_url=os.getenv("GITHUB_API_BASE_URL", "https://api.github.com"),
        github_api_version=os.getenv("GITHUB_API_VERSION", "2022-11-28"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_embeddings_model=os.getenv(
            "OPENAI_EMBEDDINGS_MODEL",
            "text-embedding-3-small",
        ),
    )
