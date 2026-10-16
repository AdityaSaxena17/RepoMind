"""Offline provider, cache and vector-space isolation regressions."""

import asyncio
import json

import httpx
import pytest
from langchain_core.documents import Document

from app import embeddings, retrieval
from app.config import QWEN_EMBEDDING_DIGEST, QWEN_QUERY_INSTRUCTION, get_settings


@pytest.mark.parametrize("query", [False, True])
@pytest.mark.parametrize(
    "failure", [None, "digest", "count", "dimensions", "zero", "nan", "http", "timeout"]
)
def test_qwen_native_transport_validation_and_query_instruction(
    monkeypatch: pytest.MonkeyPatch,
    query: bool,
    failure: str | None,
) -> None:
    clients: list[httpx.AsyncClient] = []
    requests: list[httpx.Request] = []
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert "authorization" not in request.headers
        if failure == "timeout":
            raise httpx.ReadTimeout("offline", request=request)
        if failure == "http":
            return httpx.Response(500, json={"error": "offline"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3-embedding:0.6b",
                            "digest": "changed"
                            if failure == "digest"
                            else QWEN_EMBEDDING_DIGEST,
                        }
                    ]
                },
            )
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3-embedding:0.6b"
        assert payload["truncate"] is False
        assert payload["keep_alive"] == "300s"
        assert payload["options"] == {"num_ctx": 8192}
        assert payload["input"] == (
            [f"Instruct: {QWEN_QUERY_INSTRUCTION}\nQuery:question"]
            if query
            else ["source code"]
        )
        vector = [1.0] + [0.0] * (1022 if failure == "dimensions" else 1023)
        if failure == "zero":
            vector[0] = 0.0
        elif failure == "nan":
            vector[0] = float("nan")
        return (
            httpx.Response(
                200,
                json={
                    "model": "qwen3-embedding:0.6b",
                    "embeddings": [] if failure == "count" else [vector],
                },
            )
            if failure != "nan"
            else httpx.Response(
                200,
                text=json.dumps(
                    {"model": "qwen3-embedding:0.6b", "embeddings": [vector]}
                ),
            )
        )

    def client(**kwargs: object) -> httpx.AsyncClient:
        value = original_client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(value)
        return value

    monkeypatch.setattr(embeddings.httpx, "AsyncClient", client)
    model = embeddings.QwenEmbeddings("http://127.0.0.1:11434")
    error = (
        httpx.HTTPStatusError
        if failure == "http"
        else httpx.ReadTimeout
        if failure == "timeout"
        else ValueError
    )
    if failure:
        with pytest.raises(error):
            asyncio.run(
                model.aembed_query("question")
                if query
                else model.aembed_documents(["source code"])
            )
    else:
        value = asyncio.run(
            model.aembed_query("question")
            if query
            else model.aembed_documents(["source code"])
        )
        assert len(value if query else value[0]) == 1024
    assert len(requests) == (1 if failure in {"digest", "http", "timeout"} else 2)
    assert all(value.is_closed for value in clients)


@pytest.mark.parametrize("keep_alive", [300, 120])
def test_local_embedding_factory_needs_no_provider_key(
    monkeypatch: pytest.MonkeyPatch,
    keep_alive: int,
) -> None:
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HIVE_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_EMBEDDING_KEEP_ALIVE_SECONDS", str(keep_alive))
    get_settings.cache_clear()
    try:
        model = embeddings.create_embedding_model()
        assert isinstance(model, embeddings.QwenEmbeddings)
        assert model.keep_alive_seconds == keep_alive
        assert get_settings().embedding_dimensions == 1024
        assert QWEN_EMBEDDING_DIGEST in get_settings().embedding_profile
        profile = get_settings().embedding_profile
        monkeypatch.setenv("OLLAMA_EMBEDDING_KEEP_ALIVE_SECONDS", "600")
        get_settings.cache_clear()
        assert get_settings().embedding_profile == profile
    finally:
        get_settings.cache_clear()


def test_stale_index_is_rejected_before_query_embedding() -> None:
    class Connection:
        async def fetchval(self, *_args: object) -> bool:
            return True

    class NeverEmbed:
        async def aembed_query(self, _text: str) -> list[float]:
            raise AssertionError("Stale index requested a query embedding.")

    retriever = retrieval.RepoMindParentHybridRetriever.model_construct(
        connection=Connection(),
        embeddings=NeverEmbed(),
        repository_id="stale",
    )
    with pytest.raises(ValueError, match="reindex"):
        asyncio.run(retriever._vector_candidates("question"))


@pytest.mark.parametrize("vector", [[1.0], [0.0] * 1024, [float("nan")] * 1024])
def test_index_rejects_bad_vectors_before_any_database_write(
    vector: list[float],
) -> None:
    store = retrieval.RepoMindRepositoryVectorStore(
        connection=None,
        embeddings=None,
        repository_id="test",
    )
    with pytest.raises(ValueError, match="invalid values"):
        asyncio.run(
            store.awrite_embedded_documents([Document(page_content="code")], [vector])
        )
