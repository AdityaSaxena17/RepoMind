import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from app import chat
from app.evals import execution
from app.evals.__main__ import (
    aggregate,
    compare_reports,
    grade_report,
    judge_agreement,
    write_report,
)
from app.evals.execution import CachedEmbeddings, digest, fixture_tools, load_benchmark
from app.evals.models import (
    CaseReport,
    CitationSupport,
    Claim,
    CodeContext,
    CorrectnessJudgment,
    EvaluationResult,
    EvidenceUnit,
    FactJudgment,
    GroundingJudgment,
    Question,
    ReferenceFact,
    RunReport,
    Span,
    ToolExecution,
)
from app.evals.scoring import (
    build_catalog,
    ratio,
    referenced_citations,
    score_retrieval,
    score_semantics,
    validate_grounding,
    visible_lines,
)

SHA = "a" * 40
CORPUS = {
    "app.py": "alpha = 1\nbeta = 2\ngamma = 3\ndelta = 4\n",
    "other.py": "alpha = 1\n",
}


def context(
    first: int = 1,
    last: int = 3,
    *,
    path: str = "app.py",
    index: int = 1,
    text: str | None = None,
) -> CodeContext:
    return CodeContext(
        id=f"code:{path}:{first}:{last}",
        corpus_commit_sha=SHA,
        source_index=index,
        path=path,
        start_line=first,
        end_line=last,
        content=text
        if text is not None
        else "\n".join(
            CORPUS[path].splitlines()[first - 1 : last],
        ),
    )


def unit(
    first: int = 1, last: int = 3, *, path: str = "app.py", name: str = "u1"
) -> EvidenceUnit:
    return EvidenceUnit(
        id=name,
        alternatives=[
            [
                Span(
                    path=path,
                    start_line=first,
                    end_line=last,
                    file_sha256=digest(CORPUS[path]),
                    excerpt="\n".join(CORPUS[path].splitlines()[first - 1 : last]),
                )
            ]
        ],
    )


def question(*units: EvidenceUnit, missing: bool = False) -> Question:
    return Question(
        id="test",
        category="insufficient" if missing else "single_file",
        evidence_scope="CODE",
        question="What is alpha?",
        required_evidence=list(units),
        reference_facts=[ReferenceFact(id="f1", text="alpha is 1")],
        expected_missing=["Deployment information is unavailable."] if missing else [],
        review_note="Deterministic unit-test example.",
    )


def result(
    answer: str = "alpha is 1. [1]", contexts: list[CodeContext] | None = None
) -> EvaluationResult:
    contexts = contexts if contexts is not None else [context()]
    value = EvaluationResult(
        answer=answer, final_contexts=contexts, evidence_seen=contexts
    )
    value.citation_catalog = build_catalog(value, "fixture/repo")
    return value


@pytest.mark.parametrize(
    ("windows", "recall", "precision", "overlap"),
    [
        ([context()], 1, 1, 1),
        ([context(1, 2)], 0, 0, 2 / 3),
        ([context(1, 1), context(2, 3, index=2)], 1, 0, 1),
        ([context(), context(index=2)], 1, 0.5, 1),
        ([context(1, 1, text="")], 0, 0, 0),
        ([context(1, 3, text="alpha = 1\nbeta = 2")], 0, 0, 2 / 3),
        ([], 0, 0, 0),
    ],
)
def test_retrieval_visible_union_duplicates_and_truncation(
    windows: list[CodeContext],
    recall: float,
    precision: float,
    overlap: float,
) -> None:
    metrics, overlaps = score_retrieval(
        question(unit()), result(contexts=windows), CORPUS, SHA
    )
    assert metrics["recall_at_6"].value == recall
    assert metrics["precision_at_6"].value == precision
    assert overlaps["u1"] == overlap


def test_retrieval_alternatives_and_multiple_files() -> None:
    evidence = unit()
    evidence.alternatives.append(unit(1, 1, path="other.py").alternatives[0])
    metrics, _ = score_retrieval(
        question(evidence),
        result(contexts=[context(1, 1, path="other.py")]),
        CORPUS,
        SHA,
    )
    assert metrics["recall_at_6"].value == 1
    metrics, _ = score_retrieval(
        question(unit(), unit(1, 1, path="other.py", name="u2")),
        result(),
        CORPUS,
        SHA,
    )
    assert metrics["recall_at_6"].value == 0.5
    composite = unit()
    composite.alternatives[0].extend(unit(1, 1, path="other.py").alternatives[0])
    q = question(composite)
    value = result(contexts=[context(), context(1, 1, path="other.py", index=2)])
    metrics, _ = score_retrieval(q, value, CORPUS, SHA)
    assert metrics["recall_at_6"].value == 1
    assert metrics["precision_at_6"].value == 0
    q.relevant_evidence = [unit(name="local-app"), unit(1, 1, path="other.py")]
    metrics, _ = score_retrieval(q, value, CORPUS, SHA)
    assert metrics["precision_at_6"].value == 1


def test_relevance_annotation_and_not_applicable() -> None:
    q = question(unit())
    q.relevant_evidence = [unit(4, 4, name="supportive")]
    metrics, _ = score_retrieval(q, result(contexts=[context(4, 4)]), CORPUS, SHA)
    assert metrics["recall_at_6"].value == 0
    assert metrics["precision_at_6"].value == 1
    metrics, _ = score_retrieval(question(), result(), CORPUS, SHA)
    assert all(m.status == "not_applicable" for m in metrics.values())


@pytest.mark.parametrize(
    "bad",
    [
        context(text="invented text"),
        context().model_copy(update={"corpus_commit_sha": "b" * 40}),
        context().model_copy(update={"path": "missing.py"}),
    ],
)
def test_context_verification_rejects_untrusted_metadata(bad: CodeContext) -> None:
    with pytest.raises(ValueError):
        visible_lines(bad, CORPUS, SHA)


def test_citations_use_answer_references_and_ignore_code() -> None:
    value = result("alpha [1, 99]. `[2]`\n```python\na=[3]\n```\n")
    citations = referenced_citations(value, CORPUS, SHA, "fixture/repo")
    assert [c.key for c in citations] == ["1", "99"]
    assert [c.valid for c in citations] == [True, False]
    assert not referenced_citations(result("alpha is 1."), CORPUS, SHA, "fixture/repo")


def test_citation_urls_ranges_and_tool_provenance() -> None:
    code = f"https://github.com/fixture/repo/blob/{SHA}/app.py#L1-L2"
    issue = "https://github.com/fixture/repo/issues/101"
    value = result(
        f"Code [source]({code}). Issue {issue}. {code.replace(SHA, 'b' * 40)}"
    )
    value.tool_executions = [
        ToolExecution(
            id="tool:1",
            name="issue_read",
            arguments={"issue_number": 101},
            result={"issue": {"url": issue, "body": "https://example.com/invented"}},
        )
    ]
    citations = referenced_citations(value, CORPUS, SHA, "fixture/repo")
    assert [c.valid for c in citations] == [True, True, False]
    assert citations[0].cited_excerpt == "alpha = 1\nbeta = 2"
    assert json.loads(citations[1].cited_excerpt or "null") == {
        "url": issue,
        "body": "https://example.com/invented",
    }
    value.tool_executions[0].error = "tool failed"
    assert referenced_citations(value, CORPUS, SHA, "fixture/repo")[1].valid is False
    value = result("alpha [1]", [context(text="alpha = 1")])
    assert referenced_citations(value, CORPUS, SHA, "fixture/repo")[0].valid is False


def test_plain_locations_and_orphan_citations() -> None:
    value = result("See `app.py:1-2`, app.py:1-99 and missing.py:1. [1]")
    citations = referenced_citations(value, CORPUS, SHA, "fixture/repo")
    assert [c.valid for c in citations] == [True, False, False, True]
    scores = score_semantics(
        question(),
        value,
        citations,
        GroundingJudgment(claims=[]),
        None,
    )
    assert scores["citation_precision"].value == 0


@pytest.mark.parametrize("fail", [False, True])
def test_chat_wrapper_preserves_sse_and_failure_events(
    fail: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from langchain_core.documents import Document

    writes: list[tuple[object, ...]] = []

    class FakeConnection:
        async def fetchrow(self, sql: str, *_args: object) -> dict[str, object]:
            if 'FROM "Repository"' in sql:
                return {
                    "id": "repo",
                    "owner": "fixture",
                    "name": "repo",
                    "htmlUrl": "https://github.com/fixture/repo",
                    "indexedCommitSha": SHA,
                    "defaultBranch": "main",
                    "indexStatus": "READY",
                    "indexedChunkCount": 1,
                }
            return {"content": "question"}

        async def fetch(self, *_args: object) -> list[object]:
            return []

        async def execute(self, *args: object) -> str:
            writes.append(args)
            return "OK"

    class FakeRetriever:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def ainvoke(self, _question: str) -> list[Document]:
            return [
                Document(
                    page_content="alpha = 1",
                    metadata={
                        "path": "app.py",
                        "start_line": 1,
                        "end_line": 1,
                    },
                )
            ]

    @asynccontextmanager
    async def tools(*_args: object) -> AsyncIterator[list[object]]:
        yield []

    async def generate(*_args: object) -> AsyncIterator[tuple[str, str]]:
        yield "delta", "alpha"
        if fail:
            raise RuntimeError("controlled failure")
        yield "tool", "issue_read"
        yield "delta", " is 1"

    monkeypatch.setattr(chat, "RepoMindParentHybridRetriever", FakeRetriever)
    monkeypatch.setattr(chat, "create_embedding_model", lambda: object())
    monkeypatch.setattr(chat, "scoped_mcp_tools", tools)
    monkeypatch.setattr(chat, "stream_repository_answer", generate)

    async def collect() -> list[str]:
        return [
            event
            async for event in chat.stream_repository_chat(
                FakeConnection(),
                "repo",
                "conversation",
                "user",
                "message",
            )
        ]

    events = asyncio.run(collect())
    names = [event.splitlines()[0] for event in events]
    assert names[:2] == ["event: assistant_message", "event: delta"]
    if fail:
        assert names[-1] == "event: error"
        assert '"message": "controlled failure"' in events[-1]
    else:
        assert names == [
            "event: assistant_message",
            "event: delta",
            "event: tool",
            "event: delta",
            "event: citations",
            "event: done",
        ]
        assert any("alpha is 1" in write for write in writes)


def test_semantic_scores_are_independent_and_missing_citations_do_not_pass() -> None:
    value = result("alpha is 1.")
    grounding = GroundingJudgment(
        claims=[
            Claim(
                id="claim1",
                quote="alpha is 1.",
                supported=True,
                evidence_ids=[value.evidence_seen[0].id],
                citations=[],
                citations_fully_support=False,
                reason="Supported but uncited.",
            )
        ]
    )
    correctness = CorrectnessJudgment(
        facts=[
            FactJudgment(fact_id="f1", status="contradicted", reason="Test contrast")
        ],
        material_errors=["Wrong reference fact"],
        abstention="none",
        reason="Contrast",
    )
    metrics = score_semantics(question(), value, [], grounding, correctness)
    assert metrics["groundedness"].value == 1
    assert metrics["answer_correctness"].value == 0
    assert metrics["citation_coverage"].value == 0
    assert metrics["citation_precision"].status == "not_applicable"
    assert metrics["citation_validity"].status == "not_applicable"
    unscored = score_semantics(question(), value, [], None, None)
    assert unscored["groundedness"].status == "unscored"


def test_no_claims_and_partial_abstention() -> None:
    value = result("Deployment evidence is missing.")
    judgment = CorrectnessJudgment(
        facts=[FactJudgment(fact_id="f1", status="missing", reason="Not answered")],
        material_errors=[],
        abstention="correct",
        reason="Missing deployment evidence",
    )
    scores = score_semantics(
        question(missing=True),
        value,
        [],
        GroundingJudgment(claims=[]),
        judgment,
    )
    assert scores["groundedness"].value is None
    assert scores["correct_abstention"].value == 0
    judgment.facts[0].status = "matched"
    scores = score_semantics(question(missing=True), value, [], None, judgment)
    assert scores["correct_abstention"].value == 1


def test_partial_and_joint_citations_within_one_parent_window() -> None:
    first = f"https://github.com/fixture/repo/blob/{SHA}/app.py#L1"
    second = f"https://github.com/fixture/repo/blob/{SHA}/app.py#L2"
    value = result(f"alpha precedes beta. {first} {second}")
    citations = referenced_citations(value, CORPUS, SHA, "fixture/repo")
    claim = Claim(
        id="c1",
        quote="alpha precedes beta.",
        supported=True,
        evidence_ids=[value.evidence_seen[0].id],
        citations=[{"citation_id": citations[0].id, "contributes": True}],
        citations_fully_support=False,
        reason="One line alone does not establish the order of both assignments.",
    )
    grounding = GroundingJudgment(claims=[claim])
    scores = score_semantics(question(), value, citations, grounding, None)
    assert scores["groundedness"].value == 1
    assert scores["citation_coverage"].value == 0
    assert scores["citation_precision"].value == 0.5  # Second URL is orphaned.
    claim.citations.append(
        CitationSupport(citation_id=citations[1].id, contributes=True)
    )
    claim.citations_fully_support = True
    scores = score_semantics(question(), value, citations, grounding, None)
    assert scores["citation_coverage"].value == 1
    assert scores["citation_precision"].value == 1


def test_malformed_judge_labels_are_rejected() -> None:
    value = result()
    claim = Claim(
        id="c1",
        quote="invented quote",
        supported=True,
        evidence_ids=["unknown"],
        citations=[],
        citations_fully_support=False,
        reason="Bad",
    )
    with pytest.raises(ValueError, match="quote"):
        validate_grounding(GroundingJudgment(claims=[claim]), value, [])
    claim.quote = "alpha is 1."
    with pytest.raises(ValueError, match="never seen"):
        validate_grounding(GroundingJudgment(claims=[claim]), value, [])
    with pytest.raises(ValueError, match="every reference"):
        score_semantics(
            question(),
            value,
            [],
            None,
            CorrectnessJudgment(
                facts=[],
                material_errors=[],
                abstention="none",
                reason="Omitted fact",
            ),
        )


def test_citation_support_can_use_an_alternative_consumed_source() -> None:
    value = result(
        "alpha is 1. [2]",
        [context(1, 1), context(1, 1, path="other.py", index=2)],
    )
    citations = referenced_citations(value, CORPUS, SHA, "fixture/repo")
    claim = Claim(
        id="c1",
        quote="alpha is 1.",
        supported=True,
        evidence_ids=[value.evidence_seen[0].id],
        citations=[CitationSupport(citation_id=citations[0].id, contributes=True)],
        citations_fully_support=True,
        reason="Both consumed files independently establish the same fact.",
    )
    scores = score_semantics(
        question(), value, citations, GroundingJudgment(claims=[claim]), None
    )
    assert scores["groundedness"].value == 1
    assert scores["citation_coverage"].value == 1
    assert scores["citation_precision"].value == 1


def test_versioned_dataset_spans_and_fixture_references_are_valid() -> None:
    benchmark = load_benchmark()
    assert len(benchmark.dataset.questions) == 24
    assert Counter(q.category for q in benchmark.dataset.questions) == {
        "single_file": 6,
        "architecture": 6,
        "insufficient": 4,
        "github": 4,
        "mixed": 4,
    }
    assert (
        benchmark.dataset.corpus_commit_sha
        == "678cd4ef010ef31f9293aa8ff0c92dafd255d002"
    )
    assert "app/evals/models.py" not in benchmark.corpus


def test_fixtures_match_production_defaults_and_record_misses_and_errors() -> None:
    fixtures = load_benchmark().fixtures

    async def run() -> None:
        records: list[ToolExecution] = []
        tools = await fixture_tools(fixtures, "streaming", records)
        assert {t.name for t in tools} == {
            "search_issues",
            "issue_read",
            "search_pull_requests",
            "pull_request_read",
        }
        issue_tool = next(t for t in tools if t.name == "issue_read")
        response = await issue_tool.ainvoke({"issue_number": 101})
        assert isinstance(response, dict) and "issue" in response
        await issue_tool.ainvoke({"issue_number": 999})
        assert records[0].arguments["method"] == "get"
        assert records[0].error is None
        assert records[1].fixture_miss and records[1].result == {}
        failed: list[ToolExecution] = []
        tools = await fixture_tools(fixtures, "unavailable-pr", failed)
        tool = next(t for t in tools if t.name == "pull_request_read")
        await tool.ainvoke({"pull_number": 9999})
        assert failed[0].error and not failed[0].fixture_miss
        paginated: list[ToolExecution] = []
        tools = await fixture_tools(fixtures, "parallel-proposal", paginated)
        tool = next(t for t in tools if t.name == "pull_request_read")
        await tool.ainvoke({"pull_number": 201, "per_page": 1})
        assert paginated[0].error is None and not paginated[0].fixture_miss
        await tool.ainvoke({"pull_number": 201, "per_page": 100})
        assert paginated[1].error and not paginated[1].fixture_miss

    asyncio.run(run())


def test_fixture_callables_are_accepted_by_actual_agent_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain.agents import create_agent

    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    from app.config import get_settings

    get_settings.cache_clear()

    async def construct() -> None:
        records: list[ToolExecution] = []
        tools = await fixture_tools(load_benchmark().fixtures, "streaming", records)
        agent = create_agent(chat.create_chat_model(), tools=tools)
        assert agent is not None and records == []

    try:
        asyncio.run(construct())
    finally:
        get_settings.cache_clear()


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * 1535)] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def test_embedding_cache_validates_and_reuses_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def provider() -> FakeEmbeddings:
        calls.append("provider")
        return FakeEmbeddings()

    monkeypatch.setattr(execution, "create_embedding_model", provider)
    embeddings = CachedEmbeddings(tmp_path)
    first = asyncio.run(embeddings.aembed_query("query"))
    assert asyncio.run(embeddings.aembed_query("query")) == first
    assert calls == ["provider"]
    path = next(tmp_path.glob("*.json"))
    path.write_text("[[1, 2]]", encoding="utf-8")
    with pytest.raises(ValueError, match="dimensions"):
        asyncio.run(embeddings.aembed_query("query"))


@pytest.mark.parametrize(
    "url",
    [
        "",
        "postgresql://repomind_eval:x@localhost/RepoMind",
        "postgresql://repomind:x@localhost/repomind_eval",
    ],
)
def test_database_target_rejected_before_connecting(
    url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVAL_DATABASE_URL", url)
    with pytest.raises(ValueError):
        asyncio.run(execution.connect_evaluation_database())


def test_aggregation_and_reports_keep_missing_scores_visible(tmp_path: Path) -> None:
    report = RunReport(
        metadata={"experiment_name": "test", "dataset_case_count": 3},
        cases=[
            CaseReport(
                case_id="one",
                category="code",
                question="q",
                metrics={
                    "groundedness": ratio(1, 2),
                },
            ),
            CaseReport(case_id="two", category="code", question="q"),
            CaseReport(case_id="three", category="code", question="q", status="error"),
        ],
    )
    stats = aggregate(report)["groundedness"]
    assert isinstance(stats, dict)
    assert stats["macro_average"] == 0.5 and stats["scored_cases"] == 1
    assert stats["unscored"] == 2
    write_report(report, tmp_path / "report.json")
    saved = RunReport.model_validate_json((tmp_path / "report.json").read_text())
    assert saved.metadata["complete_quality_baseline"] is False
    assert (tmp_path / "report.md").exists()
    changed = saved.model_copy(deep=True)
    changed.metadata["dataset_version"] = "different"
    with pytest.raises(ValueError, match="dataset_version"):
        compare_reports(saved, changed)


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "fact",
        "correctness",
        "abstention",
        "macro",
        "applicability",
        "answers",
        "rubric",
        "failed_calibration",
    ],
)
def test_judge_agreement_uses_fixed_inputs_and_separate_adoption_gates(
    change: str,
) -> None:
    report = RunReport(
        metadata={
            **dict.fromkeys(
                [
                    "dataset_version",
                    "fixture_version",
                    "scorer_version",
                    "dataset_sha256",
                    "fixture_sha256",
                    "corpus_commit_sha",
                    "scorer_sha256",
                    "prompt_sha256",
                    "model_configurations",
                    "retrieval_parameters",
                ],
                "same",
            ),
            "judge_configuration": {
                **dict.fromkeys(
                    [
                        "rubric_sha256",
                        "implementation_sha256",
                        "schema_sha256",
                        "calibration_sha256",
                    ],
                    "same",
                ),
                "model": "trusted",
            },
            "complete_quality_baseline": True,
            "judge_calibration": [{"passed": True}] * 12,
        },
        cases=[
            CaseReport(
                case_id="q",
                category="single_file",
                question="q",
                result=result(),
                grounding=GroundingJudgment(claims=[]),
                correctness=CorrectnessJudgment(
                    facts=[
                        FactJudgment(fact_id="f1", status="matched", reason="Verified")
                    ],
                    material_errors=[],
                    abstention="none",
                    reason="Verified",
                ),
                metrics={
                    name: ratio(1, 1)
                    for name in [
                        "groundedness",
                        "citation_precision",
                        "citation_coverage",
                        "answer_correctness",
                    ]
                },
            )
        ],
    )
    candidate = report.model_copy(deep=True)
    candidate.metadata["judge_configuration"]["model"] = "candidate"
    case = candidate.cases[0]
    if change == "fact":
        case.correctness.facts[0].status = "missing"
    if change == "correctness":
        case.metrics["answer_correctness"] = ratio(0, 1)
    if change == "abstention":
        case.correctness.abstention = "unnecessary"
    if change == "macro":
        case.metrics["citation_coverage"] = ratio(9, 10)
    if change == "applicability":
        case.metrics["groundedness"] = ratio(0, 0)
    if change == "answers":
        case.result.answer = "Different answer"
    if change == "rubric":
        candidate.metadata["judge_configuration"]["rubric_sha256"] = "changed"
    if change == "failed_calibration":
        candidate.metadata["judge_calibration"] = [{"passed": False}]
    if change in {"answers", "rubric"}:
        with pytest.raises(ValueError):
            judge_agreement(report, candidate)
    else:
        agreement = judge_agreement(report, candidate)
        assert agreement["eligible_for_adoption"] is (change == "none")
        assert agreement["requires_manual_review"] is True
    with pytest.raises(ValueError, match="judge_configuration"):
        compare_reports(report, candidate)


def test_shared_stream_preserves_text_tool_order_and_deduplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.messages import AIMessage, AIMessageChunk

    class FakeAgent:
        async def astream(self, *_args: object, **_kwargs: object):
            yield {"type": "messages", "data": (AIMessageChunk(content="hello"), {})}
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "issue_read",
                        "args": {"issue_number": 101},
                        "id": "one",
                    }
                ],
            )
            for _ in range(2):
                yield {"type": "updates", "data": {"model": {"messages": [message]}}}
            yield {"type": "messages", "data": (AIMessageChunk(content=" world"), {})}

    monkeypatch.setattr(chat, "create_agent", lambda *_args, **_kwargs: FakeAgent())
    monkeypatch.setattr(chat, "create_chat_model", lambda: object())

    async def collect() -> list[tuple[str, str]]:
        return [event async for event in chat.stream_repository_answer("q", [], [], [])]

    assert asyncio.run(collect()) == [
        ("delta", "hello"),
        ("tool", "issue_read"),
        ("delta", " world"),
    ]


@pytest.mark.parametrize("bad_attempts", [0, 1, 3])
def test_judge_tasks_keep_gold_out_of_grounding_and_bound_validation_retries(
    monkeypatch: pytest.MonkeyPatch,
    bad_attempts: int,
) -> None:
    from app.evals import judge as judge_module

    payloads: list[str] = []
    value = result("alpha is 1.")
    q = question()
    q.reference_facts[0].text = "SECRET_GOLD_FACT_FOR_TEST"

    class FakeCall:
        def __init__(self, schema: type[object]) -> None:
            self.schema = schema

        async def ainvoke(self, messages: list[tuple[str, str]]) -> object:
            payloads.append(messages[-1][1])
            if self.schema is GroundingJudgment:
                return GroundingJudgment(
                    claims=[
                        Claim(
                            id="c1",
                            quote=(
                                "invented quote"
                                if len(payloads) <= bad_attempts
                                else "alpha is 1."
                            ),
                            supported=True,
                            evidence_ids=[value.evidence_seen[0].id],
                            citations=[],
                            citations_fully_support=False,
                            reason="Visible code",
                        )
                    ]
                )
            return CorrectnessJudgment(
                facts=[FactJudgment(fact_id="f1", status="matched", reason="Test")],
                material_errors=[],
                abstention="none",
                reason="Test",
            )

    class FakeChat:
        def with_structured_output(
            self, schema: type[object], **_kwargs: object
        ) -> FakeCall:
            return FakeCall(schema)

    configuration: dict[str, object] = {}

    def create_chat(**kwargs: object) -> FakeChat:
        configuration.update(kwargs)
        return FakeChat()

    monkeypatch.setattr(judge_module, "ChatOpenAI", create_chat)
    judge = judge_module.SemanticJudge("offline-test", reasoning_effort="low")
    assert configuration["temperature"] is None
    assert configuration["reasoning"] == {"effort": "low"}
    assert judge.configuration["reasoning_effort"] == "low"
    case = CaseReport(
        case_id=q.id, category=q.category, question=q.question, result=value
    )
    if bad_attempts == 3:
        with pytest.raises(ValueError, match="quote"):
            asyncio.run(judge.grade(case, q))
        assert case.grounding is None
        assert len(payloads) == 3
    else:
        asyncio.run(judge.grade(case, q))
        assert "SECRET_GOLD_FACT_FOR_TEST" in payloads[-1]
    assert len(case.judge_validation_errors) == bad_attempts
    assert "SECRET_GOLD_FACT_FOR_TEST" not in payloads[0]


@pytest.mark.parametrize(
    "retrieval_mode,candidate_budget,expected_vector_k,expected_keyword_k",
    [
        ("hybrid", None, 24, 24),
        ("vector", None, 24, 0),
        ("keyword", None, 0, 24),
        ("hybrid", 12, 12, 12),
        ("hybrid", 24, 24, 24),
        ("vector", 24, 24, 0),
        ("keyword", 24, 0, 24),
    ],
)
def test_timeout_does_not_kill_the_remaining_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retrieval_mode: str,
    candidate_budget: int | None,
    expected_vector_k: int,
    expected_keyword_k: int,
) -> None:
    import argparse

    from app.evals import __main__ as cli

    class FakeConnection:
        async def close(self) -> None:
            return None

    async def connect() -> FakeConnection:
        return FakeConnection()

    async def verified(*_args: object) -> bool:
        return True

    class FakeRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            assert _kwargs["vector_k"] == expected_vector_k
            assert _kwargs["keyword_k"] == expected_keyword_k

        async def run(self, prompt: str) -> EvaluationResult:
            if prompt.startswith("Which"):
                await asyncio.sleep(1)
            return EvaluationResult()

    monkeypatch.setattr(cli, "connect_evaluation_database", connect)
    monkeypatch.setattr(cli, "verify_index", verified)
    monkeypatch.setattr(cli, "corpus_documents", lambda _benchmark: ("test", []))
    monkeypatch.setattr(cli, "SingleAgentRunner", FakeRunner)
    output = tmp_path / "batch.json"
    args = argparse.Namespace(
        command="retrieval",
        timeout=0.01,
        output=output,
        experiment="offline",
        dataset=execution.DATA / "dataset.json",
        fixtures=execution.DATA / "github.json",
        corpus_root=execution.ROOT,
        case=["code-01", "code-02"],
        retrieval_mode=retrieval_mode,
        candidate_budget=candidate_budget,
        rerank=False,
    )
    assert asyncio.run(cli.run(args)) == 1
    report = RunReport.model_validate_json(output.read_text())
    assert [c.status for c in report.cases] == ["error", "completed"]
    assert report.cases[0].error == "execution: TimeoutError"
    assert report.metadata["retrieval_mode"] == retrieval_mode
    assert report.metadata["retrieval_parameters"]["vector_k"] == expected_vector_k
    assert report.metadata["retrieval_parameters"]["keyword_k"] == expected_keyword_k


def test_grading_bounds_concurrency_and_continues_after_a_case_error(
    tmp_path: Path,
) -> None:
    from app.evals.__main__ import SCORER_VERSION

    active, maximum = 0, 0

    class FakeJudge:
        configuration = {"model": "offline"}

        async def calibrate(self) -> list[dict[str, object]]:
            return [{"id": "offline", "passed": True}]

        async def grade(self, case: CaseReport, q: Question) -> None:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0)
                if case.case_id == "code-01":
                    raise ValueError("Controlled judge failure")
                case.grounding = GroundingJudgment(claims=[])
                case.correctness = CorrectnessJudgment(
                    facts=[
                        FactJudgment(fact_id=f.id, status="missing", reason="Offline")
                        for f in q.reference_facts
                    ],
                    material_errors=[],
                    abstention="none",
                    reason="Offline",
                )
                case.metrics.update(
                    score_semantics(
                        q, case.result, [], case.grounding, case.correctness
                    )
                )
            finally:
                active -= 1

    report = RunReport(
        metadata={
            "mode": "full",
            "dataset_case_count": 24,
            "scorer_version": SCORER_VERSION,
            "dataset_sha256": digest(
                (execution.DATA / "dataset.json").read_text("utf-8")
            ),
            "fixture_sha256": digest(
                (execution.DATA / "github.json").read_text("utf-8")
            ),
            "scorer_sha256": digest(
                (execution.DATA.parent / "scoring.py").read_text("utf-8")
            ),
        },
        cases=[
            CaseReport(case_id=q.id, category=q.category, question=q.question)
            for q in load_benchmark().dataset.questions[:3]
        ],
    )
    asyncio.run(
        grade_report(
            report,
            FakeJudge(),
            tmp_path / "graded.json",
            execution.DATA / "dataset.json",
            execution.DATA / "github.json",
        )
    )
    assert maximum == 3 and active == 0
    assert [c.status for c in report.cases] == ["error", "completed", "completed"]
    assert report.cases[0].metrics["groundedness"].status == "error"
    assert report.metadata["complete_quality_baseline"] is False
