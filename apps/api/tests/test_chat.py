import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.chat import create_chat_model, message_text
from app.config import get_settings


def test_create_chat_model_requires_explicit_model(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
  monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
  get_settings.cache_clear()

  with pytest.raises(RuntimeError, match="OPENAI_CHAT_MODEL"):
    create_chat_model()

  get_settings.cache_clear()


def test_create_chat_model_normalizes_openai_display_name(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
  monkeypatch.setenv("OPENAI_CHAT_MODEL", "GPT-4o mini")
  get_settings.cache_clear()

  model = create_chat_model()

  assert model.model_name == "gpt-4o-mini"

  get_settings.cache_clear()


def test_message_text_reads_full_and_chunk_messages() -> None:
  assert message_text(AIMessage(content="full answer")) == "full answer"
  assert message_text(AIMessageChunk(content="streamed answer")) == "streamed answer"
  assert message_text(object()) == ""
