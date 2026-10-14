"""Opt-in database mechanics; deterministic vectors are never quality baselines."""

import asyncio
import os

import asyncpg
import pytest
from test_evals import FakeEmbeddings

from app import retrieval, worker
from app.config import get_settings
from app.evals.execution import (
    Benchmark,
    connect_evaluation_database,
    corpus_documents,
    prepare_index,
    verify_index,
)
from app.evals.models import Dataset, Fixtures, Question
from app.retrieval import RepoMindParentHybridRetriever

pytestmark = pytest.mark.skipif(
    os.getenv("REPOMIND_EVAL_INTEGRATION") != "1",
    reason="Opt in with REPOMIND_EVAL_INTEGRATION=1 and evaluation-only credentials.",
)


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_real_pgvector_indexing_retrieval_and_connection_serialization(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("OPENAI_EMBEDDINGS_MODEL", "eval-test-deterministic-1536")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", provider)
    get_settings.cache_clear()

    class DeterministicEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [
                [1.0] + [0.0] * (get_settings().embedding_dimensions - 1) for _ in texts
            ]

    embeddings = DeterministicEmbeddings()
    monkeypatch.setattr(worker, "create_embedding_model", lambda: embeddings)
    dataset = Dataset(
        version="integration",
        repository="fixture/integration",
        corpus_commit_sha="0" * 40,
        annotation_policy="Integration mechanics, not quality data.",
        questions=[
            Question(
                id="integration",
                category="single_file",
                evidence_scope="CODE",
                question="unique_integration_symbol",
                review_note="Integration mechanics",
            )
        ],
    )
    benchmark = Benchmark(
        dataset,
        Fixtures(version="integration", provenance="test", scenarios={"none": []}),
        {
            "app.py": (
                "def unique_integration_symbol():\n    return 42\n"
                "# retrieving prefixHandler\n# index embedding\n"
            ),
            "a.py": "# retrieving prefixHandler_suffix\n",
            "b.py": "# retrieving\n",
            "c.py": "# retrieving retrieving retrieving retrieving\n",
            "docs.py": "# GitHub GitHub GitHub GitHub\n",
        },
        dict.fromkeys(["app.py", "a.py", "b.py", "c.py", "docs.py"], "0" * 40),
    )

    async def check() -> None:
        connection = await connect_evaluation_database()
        repository_id, documents = corpus_documents(benchmark)
        try:
            assert (
                await connection.fetchval("SELECT current_database()")
                == "repomind_eval"
            )
            assert await connection.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname='vector'",
            )
            overlaps = await asyncio.gather(
                connection.fetchval("SELECT pg_sleep(0.02)"),
                connection.fetchval("SELECT 1"),
                return_exceptions=True,
            )
            assert any(isinstance(value, asyncpg.InterfaceError) for value in overlaps)
            assert (
                await prepare_index(connection, benchmark, embeddings) == repository_id
            )
            assert (
                await prepare_index(connection, benchmark, embeddings) == repository_id
            )
            assert await verify_index(connection, benchmark, repository_id, documents)
            assert (
                await connection.fetchval(
                    'SELECT DISTINCT vector_dims("embedding") '
                    'FROM "RepositoryChunk" WHERE "repositoryId"=$1',
                    repository_id,
                )
                == get_settings().embedding_dimensions
            )
            retriever = RepoMindParentHybridRetriever(
                connection=connection,
                embeddings=embeddings,
                repository_id=repository_id,
            )
            results = await retriever.ainvoke("unique_integration_symbol")
            assert results and results[0].metadata["path"] == "app.py"
            assert "return 42" in results[0].page_content
            vector_only = RepoMindParentHybridRetriever(
                connection=connection,
                embeddings=embeddings,
                repository_id=repository_id,
                keyword_k=0,
            )
            results = await vector_only.ainvoke("unique_integration_symbol")
            assert results and all(
                document.metadata["sources"] == ["vector"] for document in results
            )

            class NoQueryEmbeddings(FakeEmbeddings):
                async def aembed_query(self, text: str) -> list[float]:
                    raise AssertionError("Keyword-only retrieval requested embeddings")

            keyword_only = RepoMindParentHybridRetriever(
                connection=connection,
                embeddings=NoQueryEmbeddings(),
                repository_id=repository_id,
                vector_k=0,
            )
            results = await keyword_only.ainvoke("unique_integration_symbol")
            assert results and results[0].metadata["path"] == "app.py"
            assert all(
                document.metadata["sources"] == ["keyword"] for document in results
            )

            def score_parents(_query: str, passages: list[str]) -> list[float]:
                assert any(
                    "app.py\ndef unique_integration_symbol():\n    return 42" in text
                    for text in passages
                )
                assert len(passages) == len(set(passages))
                return [float("unique_integration_symbol" in text) for text in passages]

            monkeypatch.setattr(retrieval, "score_passages", score_parents)
            reranked = RepoMindParentHybridRetriever(
                connection=connection,
                embeddings=embeddings,
                repository_id=repository_id,
                rerank=True,
            )
            results = await reranked.ainvoke("unique_integration_symbol")
            assert results[0].metadata["path"] == "app.py"
            assert [doc.metadata["source_index"] for doc in results] == list(
                range(1, len(results) + 1)
            )
            # Ordinary prose must not turn the keyword branch into an all-words AND.
            rows = await retriever._keyword_candidates(
                "How does unique_integration_symbol calculate its result?"
            )
            assert rows and rows[0]["path"] == "app.py"
            assert not await retriever._keyword_candidates("the and which does")
            rows = await retriever._keyword_candidates("Explain app.py unknownword")
            assert rows and rows[0]["path"] == "app.py"
            rows = await retriever._keyword_candidates("Explain `prefixHandler`")
            # Identifier boundaries exclude the suffix.
            assert rows[0]["path"] == "app.py"
            rows = await retriever._keyword_candidates("Explain prefixHandler()")
            assert rows[0]["path"] == "app.py"
            rows = await retriever._keyword_candidates("GitHub index embedding")
            assert rows[0]["path"] == "app.py"
            assert any(row["path"] == "docs.py" for row in rows)
            rows = await retriever._keyword_candidates("`GitHub` index embedding")
            assert rows[0]["path"] == "docs.py"
            # Keep original spellings: English stemming changes 'retrieving'.
            rows = await retriever._keyword_candidates("Why retrieving?")
            assert [row["path"] for row in rows] == ["c.py", "a.py", "app.py", "b.py"]
            # Covering both terms outranks repetition of only one term.
            rows = await retriever._keyword_candidates("retrieving suffix")
            assert rows[0]["path"] == "a.py"
            await connection.execute(
                'UPDATE "RepositoryChunk" SET "embeddingProfile"=NULL '
                'WHERE "repositoryId"=$1',
                repository_id,
            )
            with pytest.raises(ValueError, match="reindex"):
                await retriever.ainvoke("unique_integration_symbol")
            assert not await verify_index(
                connection, benchmark, repository_id, documents
            )
            assert (
                await prepare_index(connection, benchmark, embeddings) == repository_id
            )
            await connection.execute(
                'UPDATE "RepositoryFile" SET "content"=$2 WHERE "repositoryId"=$1',
                repository_id,
                "corrupt corpus",
            )
            with pytest.raises(ValueError, match="frozen corpus"):
                await verify_index(connection, benchmark, repository_id, documents)
        finally:
            await connection.execute(
                'DELETE FROM "Repository" WHERE "id"=$1',
                repository_id,
            )
            await connection.close()

    try:
        asyncio.run(check())
    finally:
        get_settings.cache_clear()
