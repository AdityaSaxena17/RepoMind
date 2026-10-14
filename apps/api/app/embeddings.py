"""Local Qwen embeddings and the legacy OpenAI provider for paired experiments."""

from __future__ import annotations

import asyncio
import math

import httpx
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, FiniteFloat, JsonValue, TypeAdapter

from .config import (
    QWEN_EMBEDDING_CONTEXT,
    QWEN_EMBEDDING_DIGEST,
    QWEN_EMBEDDING_MODEL,
    QWEN_QUERY_INSTRUCTION,
    get_settings,
)


class EmbeddingResponse(BaseModel):
    model: str
    embeddings: list[list[FiniteFloat]]


class QwenEmbeddings(Embeddings):
    """Call local Ollama with pinned weights and instruction-prefixed queries."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed for synchronous callers; application paths use aembed_documents."""
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the same instruction as the asynchronous path."""
        return asyncio.run(self.aembed_query(text))

    async def aembed_query(self, text: str) -> list[float]:
        """Add Qwen's retrieval instruction only to queries, never source documents."""
        return (
            await self.aembed_documents(
                [f"Instruct: {QWEN_QUERY_INSTRUCTION}\nQuery:{text}"]
            )
        )[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed batches; reject changed weights and invalid or truncated data."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(base_url=self.base_url, timeout=120) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            tags = TypeAdapter(dict[str, JsonValue]).validate_python(response.json())
            models = tags.get("models")
            if not isinstance(models, list) or not any(
                isinstance(model, dict)
                and model.get("name") == QWEN_EMBEDDING_MODEL
                and model.get("digest") == QWEN_EMBEDDING_DIGEST
                for model in models
            ):
                raise ValueError(
                    "Pinned qwen3-embedding:0.6b weights are missing or changed. "
                    "Run ollama pull qwen3-embedding:0.6b and verify the model digest."
                )
            for offset in range(0, len(texts), 64):
                batch = texts[offset : offset + 64]
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": QWEN_EMBEDDING_MODEL,
                        "input": batch,
                        "truncate": False,
                        "options": {"num_ctx": QWEN_EMBEDDING_CONTEXT},
                    },
                )
                response.raise_for_status()
                result = EmbeddingResponse.model_validate(response.json())
                if result.model != QWEN_EMBEDDING_MODEL or len(
                    result.embeddings
                ) != len(batch):
                    raise ValueError(
                        "Ollama returned the wrong model or embedding count."
                    )
                if any(
                    len(vector) != 1024 or math.hypot(*vector) == 0
                    for vector in result.embeddings
                ):
                    raise ValueError(
                        "Qwen embeddings must be nonzero 1024-dimensional vectors."
                    )
                vectors.extend(result.embeddings)
        return vectors


def create_embedding_model() -> Embeddings:
    """Use local Qwen by default; OpenAI remains explicit for historical comparisons."""
    settings = get_settings()
    if settings.embeddings_provider == "ollama":
        return QwenEmbeddings(settings.ollama_base_url)
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for OpenAI repository embeddings."
        )
    return OpenAIEmbeddings(
        model=settings.openai_embeddings_model,
        dimensions=settings.embedding_dimensions,
        chunk_size=64,
    )
