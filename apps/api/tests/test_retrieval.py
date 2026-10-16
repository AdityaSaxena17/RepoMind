import asyncio
import copy
import sys
import threading
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app import reranking, retrieval
from app.retrieval import RepoMindParentHybridRetriever


@pytest.mark.parametrize(
    "old,new,expected",
    [
        ([("a", 1, 10)], [("a", 1, 15)], [("a", 1, 15)]),
        ([("a", 1, 10)], [("a", 5, 15)], [("a", 1, 10)]),
        (
            [("a", 1, 10), ("a", 8, 20)],
            [("a", 1, 15), ("a", 15, 25)],
            [("a", 1, 15), ("a", 15, 25)],
        ),
        (
            [("a", 1, 10), ("a", 20, 30)],
            [("a", 15, 30), ("a", 1, 5)],
            [("a", 1, 10), ("a", 20, 30)],
        ),
        (
            [("a", 1, 10), ("b", 1, 10)],
            [("a", 5, 15), ("b", 1, 15)],
            [("a", 1, 10), ("b", 1, 15)],
        ),
    ],
)
def test_parent_context_guard_retains_text_and_restores_provenance(
    old: list[tuple[str, int, int]],
    new: list[tuple[str, int, int]],
    expected: list[tuple[str, int, int]],
) -> None:
    groups = []
    for ranges, source in ((old, "vector"), (new, "keyword")):
        groups.append(
            [
                Document(
                    page_content="\n".join(
                        f"line {line}" for line in range(first, last + 1)
                    ),
                    metadata={
                        "path": path,
                        "start_line": first,
                        "end_line": last,
                        "sources": [source],
                        "score": 0.1 if source == "vector" else 0.2,
                        "source_index": index + 1,
                    },
                )
                for index, (path, first, last) in enumerate(ranges)
            ]
        )
    original, proposed = groups
    untouched = copy.deepcopy(groups)
    result = retrieval._preserve_parent_context(original, proposed)
    assert [
        (d.metadata["path"], d.metadata["start_line"], d.metadata["end_line"])
        for d in result
    ] == expected
    for index, document in enumerate(result):
        assert document == (
            proposed[index] if expected[index] == new[index] else original[index]
        )
    assert groups == untouched


def test_late_children_complete_selected_parents_with_one_fetch_and_guard() -> None:
    files = {
        path: "\n".join(f"{path} line {line}" for line in range(1, 601))
        for path in ("a.py", "b.py", "c.py")
    }
    fetches = []

    class Connection:
        async def fetch(
            self, _sql: str, repository: str, ids: list[str]
        ) -> list[dict[str, object]]:
            assert repository == "repo"
            fetches.append(ids)
            return [
                dict(id=path, path=path, content=files[path], lineCount=600)
                for path in ids
            ]

    candidates = [
        dict(
            id=f"{path}:{first}:{last}",
            file_id=path,
            path=path,
            start_line=first,
            end_line=last,
            score=0.03,
            sources={"vector" if index < 3 else "keyword"},
        )
        for index, (path, first, last) in enumerate(
            [
                ("a.py", 209, 296),
                ("a.py", 1, 104),
                ("b.py", 100, 140),
                ("a.py", 373, 446),
                ("a.py", 104, 216),
                ("b.py", 260, 280),
                ("c.py", 100, 140),
                ("b.py", 500, 550),
            ]
        )
    ]
    untouched = copy.deepcopy(candidates)
    retriever = RepoMindParentHybridRetriever.model_construct(
        connection=Connection(),
        embeddings=None,
        repository_id="repo",
        final_k=3,
    )

    async def run() -> None:
        original = await retriever._parent_documents(candidates[:3])
        fetches.clear()
        completed = await retriever._parent_documents(candidates)
        assert len(fetches) == 1
        assert [
            (d.metadata["path"], d.metadata["start_line"], d.metadata["end_line"])
            for d in completed
        ] == [("a.py", 197, 446), ("a.py", 1, 216), ("b.py", 20, 220)]
        old_lines = {
            (d.metadata["path"], line)
            for d in original
            for line in range(d.metadata["start_line"], d.metadata["end_line"] + 1)
        }
        new_lines = {
            (d.metadata["path"], line)
            for d in completed
            for line in range(d.metadata["start_line"], d.metadata["end_line"] + 1)
        }
        assert old_lines <= new_lines
        assert completed[0].metadata["sources"] == ["keyword", "vector"]
        assert (
            completed[2] == original[2]
        )  # Rollback discards rejected provenance/score.
        assert [d.metadata["source_index"] for d in completed] == [1, 2, 3]
        assert len(completed) == 3 and candidates == untouched
        for document in completed:
            first, last = document.metadata["start_line"], document.metadata["end_line"]
            assert document.page_content == "\n".join(
                files[document.metadata["path"]].splitlines()[first - 1 : last]
            )
            assert last - first + 1 <= retrieval.PARENT_MAX_LINES
            assert len(document.page_content) <= retrieval.PARENT_MAX_CHARS

    asyncio.run(run())


@pytest.mark.parametrize("late", [(200, 220), (80, 90)])
def test_late_parent_completion_cannot_clip_child_evidence(
    late: tuple[int, int],
) -> None:
    content = "\n".join(f"line {line} " + "x" * 100 for line in range(1, 601))

    class Connection:
        async def fetch(self, *_args: object) -> list[dict[str, object]]:
            return [dict(id="a", path="a", content=content, lineCount=600)]

    first, last = (100, 140) if late == (200, 220) else (150, 160)
    candidates = [
        dict(
            id=str(index),
            file_id="a",
            path="a",
            start_line=a,
            end_line=b,
            score=0.03,
            sources={source},
        )
        for index, (a, b, source) in enumerate(
            [(first, last, "vector"), (*late, "keyword")]
        )
    ]
    retriever = RepoMindParentHybridRetriever.model_construct(
        connection=Connection(),
        embeddings=None,
        repository_id="repo",
        final_k=1,
    )
    documents = asyncio.run(retriever._parent_documents(candidates))
    assert len(documents) == 1
    assert (
        documents[0].metadata["start_line"]
        <= first
        <= last
        <= documents[0].metadata["end_line"]
    )
    assert len(documents[0].page_content) <= retrieval.PARENT_MAX_CHARS
    if late == (200, 220):
        assert documents[0].metadata["end_line"] < late[0]
        assert documents[0].metadata["sources"] == ["vector"]
    else:
        assert documents[0].metadata["sources"] == ["keyword", "vector"]


def test_reranker_loads_pinned_official_model_once_without_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[dict[str, object]] = []

    class Model:
        def __init__(self, name: str, **kwargs: object) -> None:
            assert name == "Qwen/Qwen3-Reranker-0.6B"
            loads.append(kwargs)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Model)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float16="float16",
            float32="float32",
            nn=SimpleNamespace(Identity=object),
        ),
    )
    reranking.get_reranker.cache_clear()
    try:
        assert reranking.get_reranker() is reranking.get_reranker()
        assert len(loads) == 1
        assert loads[0]["revision"] == reranking.RERANKER_REVISION
        assert loads[0]["trust_remote_code"] is False
        assert loads[0]["max_length"] == 32768
        assert loads[0]["config_kwargs"] == {"use_cache": False}
        assert loads[0]["device"] == "cpu"
        assert loads[0]["prompts"] == {"query": reranking.RERANKER_INSTRUCTION}
    finally:
        reranking.get_reranker.cache_clear()


def test_reranking_scores_expanded_full_pool_off_loop_and_keeps_stable_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_thread = threading.get_ident()

    class Connection:
        async def fetch(
            self, _sql: str, _repository: str, ids: list[str]
        ) -> list[dict[str, object]]:
            assert len(ids) == 24
            return [
                dict(
                    id=str(index),
                    path=f"{index:02}.py",
                    lineCount=3,
                    content=f"prefix {index}\nchild {index}\nsurrounding {index}",
                )
                for index in range(24)
            ]

    async def candidates(_self: object, _query: str) -> list[dict[str, object]]:
        return [
            dict(
                id=str(index),
                file_id=str(index),
                path=f"{index:02}.py",
                start_line=2,
                end_line=2,
                content=f"child {index}",
                rank=index + 1,
                source="vector",
            )
            for index in range(24)
        ]

    def scores(question: str, passages: list[str]) -> list[float]:
        assert threading.get_ident() != main_thread
        assert question == "question"
        assert len(passages) == 24
        assert passages[-1] == "23.py\nprefix 23\nchild 23\nsurrounding 23"
        assert passages[0] == "00.py\nprefix 0\nchild 0\nsurrounding 0"
        return [0.0] * 23 + [1.0]

    async def duplicates(_self: object, _query: str) -> list[dict[str, object]]:
        return [
            dict(
                id="duplicate",
                file_id="0",
                path="00.py",
                start_line=2,
                end_line=2,
                rank=1,
                source="keyword",
            )
        ]

    monkeypatch.setattr(RepoMindParentHybridRetriever, "_vector_candidates", candidates)
    monkeypatch.setattr(
        RepoMindParentHybridRetriever, "_keyword_candidates", duplicates
    )
    monkeypatch.setattr(retrieval, "score_passages", scores)
    retriever = RepoMindParentHybridRetriever.model_construct(
        connection=Connection(),
        embeddings=None,
        repository_id="test",
        rerank=True,
    )
    documents = asyncio.run(retriever.ainvoke("question"))
    assert [doc.metadata["path"] for doc in documents] == [
        "23.py",
        "00.py",
        "01.py",
        "02.py",
        "03.py",
        "04.py",
    ]
    assert [doc.metadata["source_index"] for doc in documents] == list(range(1, 7))
    assert documents[1].metadata["sources"] == ["keyword", "vector"]


def test_reranker_failure_is_not_silently_replaced_with_rrf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def candidates(_self: object, _query: str) -> list[dict[str, object]]:
        return [
            dict(
                id="a",
                path="a.py",
                start_line=1,
                rank=1,
                source="vector",
                content="evidence",
            )
        ]

    def fail(_question: str, _passages: list[str]) -> list[float]:
        raise RuntimeError("model failed")

    async def parents(_self: object, _rows: object) -> list[Document]:
        return [
            Document(
                page_content="parent evidence",
                metadata={
                    "path": "a.py",
                    "start_line": 1,
                    "end_line": 1,
                    "sources": ["vector"],
                    "score": 1.0,
                },
            )
        ]

    monkeypatch.setattr(RepoMindParentHybridRetriever, "_vector_candidates", candidates)
    monkeypatch.setattr(retrieval, "score_passages", fail)
    monkeypatch.setattr(RepoMindParentHybridRetriever, "_parent_documents", parents)
    retriever = RepoMindParentHybridRetriever.model_construct(
        connection=None,
        embeddings=None,
        repository_id="test",
        keyword_k=0,
        rerank=True,
    )
    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(retriever.ainvoke("question"))


@pytest.mark.parametrize("scores", [[1.0], [float("nan"), 0.0], [0.0, float("inf")]])
def test_reranker_rejects_invalid_scores(
    scores: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        def preprocess(self, _pairs: object, **kwargs: object) -> dict[str, object]:
            assert kwargs["processing_kwargs"] == {"text": {"truncation": False}}
            return {"input_ids": SimpleNamespace(shape=(1, 5))}

        def predict(
            self, pairs: list[tuple[str, str]], **kwargs: object
        ) -> list[float]:
            assert pairs == [("query", "a"), ("query", "b")]
            assert kwargs["batch_size"] == 4
            assert kwargs["processing_kwargs"] == {"text": {"truncation": False}}
            return scores

    monkeypatch.setattr(reranking, "get_reranker", lambda: Model())
    with pytest.raises(ValueError, match="invalid relevance scores"):
        reranking.score_passages("query", ["a", "b"])


def test_empty_reranking_does_not_load_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise AssertionError("Empty retrieval must not load model weights")

    monkeypatch.setattr(reranking, "get_reranker", fail)
    assert reranking.score_passages("query", []) == []


def test_reranker_rejects_oversized_windows_instead_of_truncating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        def preprocess(self, _pairs: object, **_kwargs: object) -> dict[str, object]:
            return {"input_ids": SimpleNamespace(shape=(1, 32769))}

        def predict(self, *_args: object, **_kwargs: object) -> list[float]:
            raise AssertionError("Oversized windows must not reach inference")

    monkeypatch.setattr(reranking, "get_reranker", lambda: Model())
    with pytest.raises(ValueError, match="complete-window token limit"):
        reranking.score_passages("query", ["parent text"])


def test_parent_reranking_deduplicates_visible_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def candidates(_self: object, _query: str) -> list[dict[str, object]]:
        return [dict(id="a", path="a.py", start_line=1, source="vector", rank=1)]

    async def parents(_self: object, _rows: object) -> list[Document]:
        return [
            Document(
                page_content=text,
                metadata={
                    "path": "a.py",
                    "start_line": first,
                    "end_line": first,
                    "sources": [source],
                    "score": 1.0,
                },
            )
            for text, first, source in [
                ("one", 1, "vector"),
                ("one", 1, "keyword"),
                ("two", 100, "vector"),
                ("", 200, "vector"),
            ]
        ]

    def scores(_question: str, passages: list[str]) -> list[float]:
        assert passages == ["a.py\none", "a.py\ntwo"]
        return [0.0, 1.0]

    monkeypatch.setattr(RepoMindParentHybridRetriever, "_vector_candidates", candidates)
    monkeypatch.setattr(RepoMindParentHybridRetriever, "_parent_documents", parents)
    monkeypatch.setattr(retrieval, "score_passages", scores)
    retriever = RepoMindParentHybridRetriever.model_construct(
        connection=None,
        embeddings=None,
        repository_id="test",
        keyword_k=0,
        rerank=True,
    )
    docs = asyncio.run(retriever.ainvoke("query"))
    assert [doc.page_content for doc in docs] == ["two", "one"]
    assert docs[1].metadata["sources"] == ["keyword", "vector"]


def test_parent_reranking_uses_trimmed_text_and_distinct_same_file_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [f"line {index}: café " + "x" * 100 for index in range(1, 501)]

    class Connection:
        async def fetch(self, *_args: object) -> list[dict[str, object]]:
            return [
                dict(id="file", path="a.py", lineCount=500, content="\n".join(lines))
            ]

    async def candidates(_self: object, _query: str) -> list[dict[str, object]]:
        return [
            dict(
                id=str(index),
                file_id="file",
                path="a.py",
                start_line=line,
                end_line=line,
                content=lines[line - 1],
                source="vector",
                rank=index,
            )
            for index, line in enumerate([100, 400], start=1)
        ]

    def scores(_question: str, passages: list[str]) -> list[float]:
        assert len(passages) == 2
        assert all(
            len(passage) <= retrieval.PARENT_MAX_CHARS + len("a.py\n")
            for passage in passages
        )
        assert all(passage.count("\n") > 100 for passage in passages)
        return [0.0, 1.0]

    monkeypatch.setattr(RepoMindParentHybridRetriever, "_vector_candidates", candidates)
    monkeypatch.setattr(retrieval, "score_passages", scores)
    retriever = RepoMindParentHybridRetriever.model_construct(
        connection=Connection(),
        embeddings=None,
        repository_id="test",
        keyword_k=0,
        rerank=True,
    )
    docs = asyncio.run(retriever.ainvoke("query"))
    assert len(docs) == 2
    assert docs[0].metadata["start_line"] > docs[1].metadata["end_line"]
    for doc in docs:
        start, end = doc.metadata["start_line"], doc.metadata["end_line"]
        assert doc.page_content == "\n".join(lines[start - 1 : end])
        assert len(doc.page_content.splitlines()) == end - start + 1


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
