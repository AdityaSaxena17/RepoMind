import json
import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from arq.connections import RedisSettings

QWEN_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
QWEN_EMBEDDING_DIGEST = (
    "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
)
QWEN_EMBEDDING_CONTEXT = 8192
QWEN_QUERY_INSTRUCTION = (
    "Given a question about a code repository, retrieve relevant code passages "
    "that answer the question."
)


@dataclass(frozen=True)
class Settings:
    redis_host: str
    redis_port: int
    redis_database: int
    arq_queue_name: str
    database_url: str
    internal_api_credential: str | None
    github_token: str | None
    hive_api_key: str | None
    hive_chat_model: str
    openai_api_key: str | None
    openai_embeddings_model: str
    openai_chat_model: str | None
    embeddings_provider: str
    ollama_base_url: str
    repository_rerank: bool
    agent_tool_call_limit: int = 6
    agent_model_call_limit: int = 8
    agent_timeout_seconds: float = 120.0
    code_context_cache_ttl_seconds: int = 900
    synthesis_reasoning_effort: str = "high"
    ollama_embedding_keep_alive_seconds: int = 300

    @property
    def embedding_model(self) -> str:
        """Identify the configured model independently of the chat provider."""
        return (
            QWEN_EMBEDDING_MODEL
            if self.embeddings_provider == "ollama"
            else self.openai_embeddings_model
        )

    @property
    def embedding_dimensions(self) -> int:
        """Return the native output dimension used by the configured provider."""
        return 1024 if self.embeddings_provider == "ollama" else 1536

    @property
    def embedding_profile(self) -> str:
        """Fingerprint the vector space, including Qwen's weights and query prompt."""
        profile = {
            "provider": self.embeddings_provider,
            "model": self.embedding_model,
            "dimensions": self.embedding_dimensions,
        }
        if self.embeddings_provider == "ollama":
            profile.update(
                digest=QWEN_EMBEDDING_DIGEST,
                query_instruction=QWEN_QUERY_INSTRUCTION,
                num_ctx=QWEN_EMBEDDING_CONTEXT,
                truncate=False,
            )
        return json.dumps(profile, sort_keys=True)

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


def read_optional_token(value: str | None) -> str | None:
    """Return a configured token unless it looks like an empty placeholder."""
    if value is None:
        return None

    token = value.strip()
    lower = token.lower()

    if (
        not token
        or "your" in lower
        or "replace" in lower
        or "example" in lower
        or "optional" in lower
    ):
        return None

    return token


@lru_cache
def get_settings() -> Settings:
    tool_limit = int(os.getenv("AGENT_TOOL_CALL_LIMIT", "6"))
    model_limit = int(os.getenv("AGENT_MODEL_CALL_LIMIT", "8"))
    agent_timeout = float(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))
    cache_ttl = int(os.getenv("CODE_CONTEXT_CACHE_TTL_SECONDS", "900"))
    synthesis_effort = os.getenv("SYNTHESIS_REASONING_EFFORT", "high")
    if synthesis_effort not in {"none", "low", "high", "max"}:
        raise ValueError("SYNTHESIS_REASONING_EFFORT must be none, low, high or max.")
    embedding_keep_alive = int(os.getenv("OLLAMA_EMBEDDING_KEEP_ALIVE_SECONDS", "300"))
    if embedding_keep_alive < 1:
        raise ValueError("OLLAMA_EMBEDDING_KEEP_ALIVE_SECONDS must be positive.")
    if tool_limit < 1 or model_limit < 1:
        raise ValueError("Agent tool and model call limits must be positive.")
    if not 0 < agent_timeout < float("inf") or cache_ttl < 0:
        raise ValueError(
            "Agent timeout must be finite/positive; cache TTL nonnegative."
        )
    embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", "ollama")
    if embeddings_provider not in {"ollama", "openai"}:
        raise ValueError("EMBEDDINGS_PROVIDER must be ollama or openai.")
    reranker = os.getenv("REPOSITORY_RERANKER", "none")
    if reranker not in {"none", "qwen3"}:
        raise ValueError("REPOSITORY_RERANKER must be none or qwen3.")
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
        github_token=read_optional_token(
            os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"),
        ),
        hive_api_key=read_optional_token(os.getenv("HIVE_API_KEY")),
        hive_chat_model=(
            os.getenv("HIVE_CHAT_MODEL", "deepseek-ai/deepseek-v4.1-flash").strip()
            or "deepseek-ai/deepseek-v4.1-flash"
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_embeddings_model=os.getenv(
            "OPENAI_EMBEDDINGS_MODEL",
            "text-embedding-3-small",
        ),
        openai_chat_model=normalize_openai_chat_model(os.getenv("OPENAI_CHAT_MODEL")),
        embeddings_provider=embeddings_provider,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        repository_rerank=reranker == "qwen3",
        agent_tool_call_limit=tool_limit,
        agent_model_call_limit=model_limit,
        agent_timeout_seconds=agent_timeout,
        code_context_cache_ttl_seconds=cache_ttl,
        synthesis_reasoning_effort=synthesis_effort,
        ollama_embedding_keep_alive_seconds=embedding_keep_alive,
    )
