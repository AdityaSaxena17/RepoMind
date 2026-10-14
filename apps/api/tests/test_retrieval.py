import asyncio

import pytest
from langchain_core.documents import Document

from app.retrieval import RepoMindParentHybridRetriever


@pytest.mark.parametrize("source", ["vector", "keyword"])
def test_single_branch_retrieval_skips_disabled_search(
    source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def vector(_self: object, _query: str) -> list[dict[str, object]]:
        assert source == "vector", "Disabled vector branch invoked"
        calls.append("vector")
        return [dict(id="child", path="app.py", start_line=1, rank=1, source="vector")]

    async def keyword(_self: object, _query: str) -> list[dict[str, object]]:
        assert source == "keyword", "Disabled keyword branch invoked"
        calls.append("keyword")
        return [dict(id="child", path="app.py", start_line=1, rank=1, source="keyword")]

    async def parents(
        _self: object,
        candidates: list[dict[str, object]],
    ) -> list[Document]:
        assert len(candidates) == 1 and candidates[0]["sources"] == {source}
        return [Document(page_content="parent evidence")]

    monkeypatch.setattr(RepoMindParentHybridRetriever, "_vector_candidates", vector)
    monkeypatch.setattr(RepoMindParentHybridRetriever, "_keyword_candidates", keyword)
    monkeypatch.setattr(RepoMindParentHybridRetriever, "_parent_documents", parents)
    retriever = RepoMindParentHybridRetriever.model_construct(
        connection=None,
        embeddings=None,
        repository_id="ablation",
        vector_k=12 if source == "vector" else 0,
        keyword_k=12 if source == "keyword" else 0,
    )
    documents = asyncio.run(retriever.ainvoke("question"))
    assert [document.page_content for document in documents] == ["parent evidence"]
    assert calls == [source]


def test_keyword_references_are_parameterized_and_preserve_code_names() -> None:
    calls: list[tuple[object, ...]] = []

    class Connection:
        async def fetch(self, sql: str, *arguments: object) -> list[dict[str, object]]:
            assert "ExactName" not in sql and "OR 1=1" not in sql
            calls.append(arguments)
            return []

    retriever = RepoMindParentHybridRetriever.model_construct(
        connection=Connection(), embeddings=None, repository_id="test"
    )
    question = (
        "Explain `ExactName` and _aget_relevant_documents in worker.py `symbol` OR 1=1"
    )
    assert asyncio.run(retriever._keyword_candidates(question)) == []
    assert calls[0][:3] == ("test", question, 24)
    assert calls[0][4] == ["worker.py"]
    assert calls[0][5] == ["ExactName", "_aget_relevant_documents", "symbol"]
    assert asyncio.run(retriever._keyword_candidates("  ")) == []
    assert len(calls) == 1


@pytest.mark.parametrize(
    "query,identifiers",
    [
        ("Explain GitHub and RepoMind indexing", []),
        ("Explain prefixHandler", []),
        ("Explain `prefixHandler`", ["prefixHandler"]),
        ("Explain prefixHandler() and search_code", ["prefixHandler", "search_code"]),
        ("How is READY set?", []),
        ("Explain `READY`", ["READY"]),
    ],
)
def test_identifier_boosts_require_explicit_code_syntax(
    query: str,
    identifiers: list[str],
) -> None:
    class Connection:
        async def fetch(self, _sql: str, *arguments: object) -> list[dict[str, object]]:
            assert arguments[1] == query  # Ordinary terms remain searchable.
            assert arguments[5] == identifiers
            return []

    retriever = RepoMindParentHybridRetriever.model_construct(
        connection=Connection(),
        embeddings=None,
        repository_id="test",
    )
    assert asyncio.run(retriever._keyword_candidates(query)) == []


def test_fusion_ties_have_stable_source_order() -> None:
    a = dict(
        id="a",
        file_id="a",
        path="a.py",
        start_line=1,
        end_line=2,
        source="vector",
        rank=1,
    )
    b = dict(
        id="b",
        file_id="b",
        path="b.py",
        start_line=1,
        end_line=2,
        source="keyword",
        rank=1,
    )
    for first, second in [(a, b), (b, a)]:
        results = RepoMindParentHybridRetriever._fuse_candidates([first], [second])
        assert [row["id"] for row in results] == ["a", "b"]


def test_hybrid_queries_do_not_overlap_on_shared_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = False
    completed: list[str] = []

    async def query(_self: object, _question: str) -> list[dict[str, object]]:
        nonlocal active
        if active:
            raise RuntimeError("another operation is in progress")
        active = True
        await asyncio.sleep(0)
        active = False
        completed.append("query")
        return []

    monkeypatch.setattr(RepoMindParentHybridRetriever, "_vector_candidates", query)
    monkeypatch.setattr(RepoMindParentHybridRetriever, "_keyword_candidates", query)
    retriever = RepoMindParentHybridRetriever.model_construct(
        connection=None,
        embeddings=None,
        repository_id="regression",
    )
    assert asyncio.run(retriever.ainvoke("indexing")) == []
    assert completed == ["query", "query"]


def test_parent_hybrid_retriever_fuses_vector_and_keyword_ranks() -> None:
    fused = RepoMindParentHybridRetriever._fuse_candidates(
        [
            {
                "id": "chunk_1",
                "file_id": "file_1",
                "path": "app.py",
                "start_line": 1,
                "end_line": 5,
                "source": "vector",
                "rank": 1,
            },
        ],
        [
            {
                "id": "chunk_1",
                "file_id": "file_1",
                "path": "app.py",
                "start_line": 1,
                "end_line": 5,
                "source": "keyword",
                "rank": 1,
            },
            {
                "id": "chunk_2",
                "file_id": "file_2",
                "path": "lib.py",
                "start_line": 1,
                "end_line": 5,
                "source": "keyword",
                "rank": 2,
            },
        ],
    )

    assert [item["id"] for item in fused] == ["chunk_1", "chunk_2"]
    assert fused[0]["sources"] == {"vector", "keyword"}
