from app.config import normalize_asyncpg_database_url, normalize_openai_chat_model


def test_normalize_asyncpg_database_url_removes_prisma_schema_param() -> None:
  assert (
    normalize_asyncpg_database_url(
      "postgresql://user:pass@localhost:5433/RepoMind?schema=public",
    )
    == "postgresql://user:pass@localhost:5433/RepoMind"
  )


def test_normalize_asyncpg_database_url_keeps_non_prisma_query_params() -> None:
  assert (
    normalize_asyncpg_database_url(
      "postgresql://user:pass@localhost:5433/RepoMind?sslmode=require&schema=public",
    )
    == "postgresql://user:pass@localhost:5433/RepoMind?sslmode=require"
  )


def test_normalize_openai_chat_model_accepts_api_model_id() -> None:
  assert normalize_openai_chat_model("gpt-4o-mini") == "gpt-4o-mini"


def test_normalize_openai_chat_model_converts_display_label() -> None:
  assert normalize_openai_chat_model("GPT-4o mini") == "gpt-4o-mini"


def test_normalize_openai_chat_model_treats_blank_as_missing() -> None:
  assert normalize_openai_chat_model("  ") is None
