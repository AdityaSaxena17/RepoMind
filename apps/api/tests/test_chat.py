import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.chat import create_chat_model, message_text
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


def test_message_text_reads_full_and_chunk_messages() -> None:
    assert message_text(AIMessage(content="full answer")) == "full answer"
    assert message_text(AIMessageChunk(content="streamed answer")) == "streamed answer"
    assert message_text(object()) == ""


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


@pytest.mark.parametrize("failure", [None, "auth", "timeout"])
def test_hive_agent_streams_tool_round_trip_and_surfaces_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    import asyncio
    import json

    import httpx
    from langchain_core.documents import Document
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from openai import APITimeoutError, AuthenticationError

    from app import chat

    monkeypatch.setenv("HIVE_API_KEY", "test-hive-key")
    monkeypatch.setenv("HIVE_CHAT_MODEL", "deepseek-ai/deepseek-v4.1-flash")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-must-not-be-used")
    requests: list[dict[str, object]] = []
    tool_calls: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.thehive.ai/api/v3/chat/completions"
        assert request.headers["authorization"] == "Bearer test-hive-key"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-ai/deepseek-v4.1-flash"
        assert payload["stream"] is True
        assert payload["reasoning_effort"] == "high"
        assert payload["max_completion_tokens"] == 16384
        assert payload["max_thinking_tokens"] == 8192
        assert "stream_options" not in payload
        assert payload["tools"][0]["function"]["name"] == "issue_read"
        requests.append(payload)
        if failure == "auth":
            return httpx.Response(401, json={"error": {"message": "offline"}})
        if failure == "timeout":
            raise httpx.ReadTimeout("offline", request=request)
        if len(requests) == 1:
            assert "[1] app/example.py:1-1" in payload["messages"][-1]["content"]
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_issue",
                        "type": "function",
                        "function": {
                            "name": "issue_read",
                            "arguments": '{"issue_number": 42}',
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            assert payload["messages"][-2]["tool_calls"][0]["id"] == "call_issue"
            assert payload["messages"][-1]["role"] == "tool"
            assert (
                "https://github.com/example/repo/issues/42"
                in payload["messages"][-1]["content"]
            )
            delta = {"role": "assistant", "content": "The value is 7 [1]."}
            finish_reason = "stop"
        chunk = {
            "id": "offline",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": payload["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
        )

    async def issue_read(issue_number: int) -> dict[str, str]:
        tool_calls.append(issue_number)
        return {"url": "https://github.com/example/repo/issues/42", "title": "Example"}

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

            def create_model(**kwargs: object) -> ChatOpenAI:
                return ChatOpenAI(http_async_client=client, **kwargs)

            monkeypatch.setattr(chat, "ChatOpenAI", create_model)
            tools = [
                StructuredTool.from_function(
                    coroutine=issue_read,
                    name="issue_read",
                    description="Read an issue.",
                )
            ]
            documents = [
                Document(
                    page_content="VALUE = 7",
                    metadata={"path": "app/example.py", "start_line": 1, "end_line": 1},
                )
            ]

            async def collect() -> list[tuple[str, str]]:
                return [
                    event
                    async for event in chat.stream_repository_answer(
                        "What is the value and issue 42?",
                        [],
                        documents,
                        tools,
                    )
                ]

            if failure:
                with pytest.raises(
                    AuthenticationError if failure == "auth" else APITimeoutError
                ):
                    await collect()
            else:
                assert await collect() == [
                    ("tool", "issue_read"),
                    ("delta", "The value is 7 [1]."),
                ]

    get_settings.cache_clear()
    try:
        asyncio.run(run())
    finally:
        get_settings.cache_clear()
    assert len(requests) == (1 if failure == "auth" else 2)
    assert tool_calls == ([42] if failure is None else [])
