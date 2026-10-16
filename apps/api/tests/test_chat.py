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


@pytest.mark.parametrize("effort", [None, "none", "low", "high", "max"])
def test_synthesis_reasoning_changes_only_the_hive_synthesis_role(
    monkeypatch: pytest.MonkeyPatch, effort: str | None
) -> None:
    monkeypatch.setenv("HIVE_API_KEY", "offline")
    if effort is None:
        monkeypatch.delenv("SYNTHESIS_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("SYNTHESIS_REASONING_EFFORT", effort)
    get_settings.cache_clear()
    try:
        specialist = create_chat_model()
        synthesis = create_chat_model(purpose="synthesis")
        assert synthesis.reasoning_effort == (effort or "high")
        assert specialist.reasoning_effort == "high"
        assert create_chat_model(purpose="router").reasoning_effort == "none"
        specialist_params = specialist._default_params.copy()
        synthesis_params = synthesis._default_params.copy()
        specialist_params.pop("reasoning_effort")
        synthesis_params.pop("reasoning_effort")
        assert synthesis_params == specialist_params
        assert synthesis.request_timeout == specialist.request_timeout == 120
        assert synthesis.max_retries == specialist.max_retries == 1
    finally:
        get_settings.cache_clear()


def test_synthesis_preserves_nonreasoning_openai_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIVE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("SYNTHESIS_REASONING_EFFORT", "low")
    get_settings.cache_clear()
    try:
        synthesis = create_chat_model(purpose="synthesis")
        assert synthesis._default_params == create_chat_model()._default_params
        assert synthesis.reasoning_effort is None
        assert synthesis.use_responses_api is True
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "name,value",
    [
        ("SYNTHESIS_REASONING_EFFORT", "medium"),
        ("OLLAMA_EMBEDDING_KEEP_ALIVE_SECONDS", "0"),
        ("OLLAMA_EMBEDDING_KEEP_ALIVE_SECONDS", "-1"),
        ("OLLAMA_EMBEDDING_KEEP_ALIVE_SECONDS", "inf"),
    ],
)
def test_latency_settings_reject_unsupported_or_unbounded_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError):
            get_settings()
    finally:
        get_settings.cache_clear()
