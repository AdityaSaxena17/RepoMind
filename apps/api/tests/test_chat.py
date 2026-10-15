import pytest

from app.chat import create_chat_model
from app.config import get_settings


def test_create_chat_model_requires_explicit_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("HIVE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="OPENAI_CHAT_MODEL"):
        create_chat_model()

    get_settings.cache_clear()


def test_create_chat_model_normalizes_openai_display_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("HIVE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "GPT-4o mini")
    get_settings.cache_clear()

    model = create_chat_model()

    assert model.model_name == "gpt-4o-mini"

    get_settings.cache_clear()


@pytest.mark.parametrize("hive", [False, True])
def test_router_model_preserves_provider_selection_and_bounds_requests(
    hive: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    if hive:
        monkeypatch.setenv("HIVE_API_KEY", "test-hive-key")
    else:
        monkeypatch.delenv("HIVE_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        model = create_chat_model(purpose="router")
        assert model.model_name == (
            get_settings().hive_chat_model if hive else "gpt-4o-mini"
        )
        assert model.streaming is False
        assert model.disable_streaming is True
        assert model.use_responses_api is False
        assert model.stream_usage is False
        assert model.max_tokens == 512
        assert model.request_timeout == 15
        assert model.max_retries == 0
        assert model.extra_body is None
        assert model.temperature == 0
        assert model.reasoning_effort == ("none" if hive else None)
    finally:
        get_settings.cache_clear()


def test_hive_chat_uses_the_shared_key_without_openai_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVE_API_KEY", "test-hive-key")
    monkeypatch.delenv("HIVE_CHAT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    get_settings.cache_clear()
    try:
        model = create_chat_model()
        assert model.model_name == "deepseek-ai/deepseek-v4.1-flash"
        assert model.openai_api_base == "https://api.thehive.ai/api/v3"
        assert model.openai_api_key is not None
        assert model.openai_api_key.get_secret_value() == "test-hive-key"
        assert model.use_responses_api is False
        assert model.stream_usage is False
        assert model.reasoning_effort == "high"
        assert model.max_tokens == 16384
        assert model.extra_body == {"max_thinking_tokens": 8192}
        assert model.request_timeout == 120
        assert model.max_retries == 1
    finally:
        get_settings.cache_clear()
