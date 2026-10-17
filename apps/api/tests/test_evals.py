import asyncio
import json
from collections import Counter
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

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
    GroundingBatch,
    GroundingJudgment,
    GroundingSection,
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


def test_execution_timing_tracks_overlapping_calls_and_existing_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    metrics = execution._ExecutionMetrics()
    code, github, tool = uuid4(), uuid4(), uuid4()

    async def run() -> None:
        clock[0] = 101.0
        await metrics.on_chat_model_start(
            {},
            [[HumanMessage(content="private")]],
            run_id=code,
            metadata={"repomind_role": "CODE", "repomind_phase": "investigation"},
        )
        clock[0] = 102.0
        await metrics.on_chat_model_start(
            {}, [], run_id=github, metadata={"repomind_role": "GITHUB"}
        )
        clock[0] = 103.0
        await metrics.on_tool_start(
            {"name": "issue_read"},
            "private arguments",
            run_id=tool,
            parent_run_id=github,
            metadata={"repomind_role": "GITHUB"},
        )
        await metrics.on_llm_new_token("", run_id=code)
        clock[0] = 104.0
        await metrics.on_llm_new_token("private", run_id=code)
        await metrics.on_llm_new_token("private", run_id=github)
        clock[0] = 105.0
        await metrics.on_tool_end(
            ToolMessage(content="private output", tool_call_id="x"), run_id=tool
        )
        response = LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="private",
                            usage_metadata={
                                "input_tokens": 4,
                                "output_tokens": 2,
                                "total_tokens": 6,
                            },
                        )
                    )
                ]
            ]
        )
        clock[0] = 106.0
        await metrics.on_llm_end(response, run_id=github)
        clock[0] = 108.0
        await metrics.on_custom_event(
            "repomind_answer_delta", {"text": "private", "native": False}
        )
        assert metrics.first_delta is None
        await metrics.on_custom_event("repomind_answer_delta", {"text": "private"})
        await metrics.on_llm_end(response, run_id=code)
        await metrics.on_llm_new_token("private", run_id=code)

    asyncio.run(run())
    by_id = {r["run_id"]: r for r in metrics.timings}
    assert by_id[str(code)]["duration_seconds"] == 7
    assert by_id[str(code)]["first_token_seconds"] == 3
    assert by_id[str(code)]["message_characters"] == len("private")
    assert by_id[str(code)]["phase"] == "investigation"
    assert by_id[str(code)]["output_kind"] == "completion"
    assert by_id[str(code)]["usage"] == metrics.usage[:1]
    assert by_id[str(github)]["duration_seconds"] == 4
    assert by_id[str(github)]["first_token_seconds"] == 2
    assert by_id[str(tool)] == {
        "run_id": str(tool),
        "parent_run_id": str(github),
        "kind": "tool",
        "role": "GITHUB",
        "tool_name": "issue_read",
        "started_seconds": 3.0,
        "duration_seconds": 2.0,
        "outcome": "success",
    }
    assert metrics.first_delta == 8 and metrics.completed_model_calls == 2
    assert (
        metrics.usage
        == [{"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}] * 2
    )
    assert "private" not in json.dumps(metrics.timings)
    assert metrics._active == {}


@pytest.mark.parametrize("kind", ["model", "tool", "retriever"])
@pytest.mark.parametrize("outcome", ["error", "canceled", "interrupted"])
def test_execution_timing_records_failures_and_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    outcome: str,
) -> None:
    clock = [10.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    metrics = execution._ExecutionMetrics()
    run_id = uuid4()

    async def run() -> None:
        if kind == "model":
            await metrics.on_chat_model_start(
                {}, [], run_id=run_id, metadata={"langgraph_node": "query_router"}
            )
        elif kind == "tool":
            await metrics.on_tool_start(
                {"name": "read"},
                "private",
                run_id=run_id,
                metadata={"langgraph_node": "code_agent"},
            )
        else:
            await metrics.on_retriever_start(
                {},
                "private",
                run_id=run_id,
                metadata={"langgraph_node": "code_agent"},
            )
        clock[0] = 12.5
        error = (
            asyncio.CancelledError("private")
            if outcome != "error"
            else ValueError("private")
        )
        if outcome == "interrupted":
            metrics.finish_pending("canceled")
        elif kind == "model":
            await metrics.on_llm_error(error, run_id=run_id)
        elif kind == "tool":
            await metrics.on_tool_error(error, run_id=run_id)
        else:
            await metrics.on_retriever_error(error, run_id=run_id)
        # A late native cancellation callback must not duplicate a finalized run.
        await metrics.on_llm_error(error, run_id=run_id)

    asyncio.run(run())
    assert len(metrics.timings) == 1
    assert metrics.timings[0]["outcome"] == (
        "error" if outcome == "error" else "canceled"
    )
    assert metrics.timings[0]["duration_seconds"] == 2.5
    assert metrics.timings[0]["role"] == ("ROUTER" if kind == "model" else "CODE")
    assert metrics.completed_model_calls == 0 and not metrics.usage
    assert "private" not in json.dumps(metrics.timings)
    assert not metrics._active


def test_execution_timing_recognizes_failed_tool_messages() -> None:
    metrics = execution._ExecutionMetrics()
    run_id = uuid4()

    async def run() -> None:
        await metrics.on_tool_start({"name": "read"}, "private", run_id=run_id)
        await metrics.on_tool_end(
            ToolMessage(content="private failure", tool_call_id="x", status="error"),
            run_id=run_id,
        )

    asyncio.run(run())
    assert metrics.timings[0]["outcome"] == "error"
    assert metrics.timings[0]["role"] == "UNKNOWN"


def test_retrieval_timing_is_nested_and_records_no_query_or_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [10.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    metrics = execution._ExecutionMetrics()
    tool, retrieval = uuid4(), uuid4()

    async def run() -> None:
        await metrics.on_tool_start(
            {"name": "search_repository_code"},
            "private",
            run_id=tool,
            metadata={"repomind_role": "CODE"},
        )
        clock[0] = 11.0
        await metrics.on_retriever_start(
            {},
            "private",
            run_id=retrieval,
            parent_run_id=tool,
            metadata={"repomind_role": "CODE"},
        )
        clock[0] = 12.0
        await metrics.on_retriever_end([], run_id=retrieval)
        clock[0] = 13.0
        await metrics.on_tool_end("private", run_id=tool)

    asyncio.run(run())
    assert metrics.timings[0]["kind"] == "retriever"
    assert metrics.timings[0]["parent_run_id"] == str(tool)
    assert metrics.timings[0]["duration_seconds"] == 1
    assert metrics.timings[1]["duration_seconds"] == 3
    assert "private" not in json.dumps(metrics.timings)


@pytest.mark.parametrize("eligible", [False, True])
def test_fixture_catalog_uses_metadata_and_preserves_defaults_and_records(
    monkeypatch: pytest.MonkeyPatch, eligible: bool
) -> None:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    server = FastMCP("fixture-policy")

    @server.tool(
        meta={
            "repomind": {
                "capability": "github.read" if eligible else "github.write",
                "repository_scoped": True,
            }
        },
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    async def dynamic_reader(
        issue_number: int, page: int = 1, per_page: int = 20
    ) -> dict[str, int]:
        """New repository-scoped reader."""
        pytest.fail("The fixture adapter must not execute the real provider.")

    monkeypatch.setattr(execution, "mcp", server)

    async def run() -> None:
        records: list[ToolExecution] = []
        tools = await fixture_tools(load_benchmark().fixtures, "streaming", records)
        assert [tool.name for tool in tools] == (["dynamic_reader"] if eligible else [])
        if eligible:
            definition = (await server.list_tools())[0]
            assert tools[0].args_schema == definition.inputSchema
            assert tools[0].description == definition.description
            await tools[0].ainvoke({"issue_number": 1})
            assert len(records) == 1
            assert records[0].name == "dynamic_reader"
            assert records[0].arguments == {
                "issue_number": 1,
                "page": 1,
                "per_page": 20,
            }
            assert records[0].fixture_miss is True
        else:
            assert records == []

    asyncio.run(run())


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


@pytest.mark.parametrize("comments_first", [False, True])
def test_shared_url_citations_use_the_catalogs_observed_resource(
    comments_first: bool,
) -> None:
    url = "https://github.com/fixture/repo/issues/101"
    comment = url + "#issuecomment-19"
    details = ToolExecution(
        id="tool:details",
        name="issue_read",
        arguments={"issue_number": 101},
        result={"issue": {"url": url, "state": "open"}},
    )
    comments = ToolExecution(
        id="tool:comments",
        name="issue_read",
        arguments={"issue_number": 101, "method": "get_comments"},
        result={"url": url, "comments": [{"url": comment, "body": "Defer."}]},
    )
    value = result(f"Open ({url}); defer ({comment}).")
    value.tool_executions = (
        [comments, details] if comments_first else [details, comments]
    )
    value.citation_catalog.update({url: details.id, comment: comments.id})
    citations = referenced_citations(value, CORPUS, SHA, "fixture/repo")
    assert [c.evidence_id for c in citations] == [details.id, comments.id]
    assert all(c.valid for c in citations)
    assert json.loads(citations[0].cited_excerpt or "null")["state"] == "open"
    assert json.loads(citations[1].cited_excerpt or "null")["body"] == "Defer."


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


@pytest.mark.parametrize("fault", [None, "repository", "number", "error", "method"])
def test_pr_url_provenance_includes_only_matching_successful_reads(
    fault: str | None,
) -> None:
    url = "https://github.com/fixture/repo/pull/17"
    value = result(f"The patch changes app.py ({url}).")
    details = ToolExecution(
        id="tool:details",
        name="pull_request_read",
        arguments={"pull_number": 17, "method": "get"},
        result={
            "repository": "fixture/repo",
            "pull_request": {"number": 17, "url": url},
        },
    )
    diff = ToolExecution(
        id="tool:diff",
        name="pull_request_read",
        arguments={"pull_number": 17, "method": "get_diff"},
        result={
            "repository": "fixture/repo",
            "pull_number": 17,
            "diff": "-before\n+after",
        },
    )
    if fault == "repository":
        diff.result["repository"] = "other/repo"
    elif fault == "number":
        diff.result["pull_number"] = 18
    elif fault == "error":
        diff.error = "Unavailable."
    elif fault == "method":
        diff.arguments["method"] = "get_comments"
    value.tool_executions = [details, diff]
    value.citation_catalog[url] = details.id
    citation = referenced_citations(value, CORPUS, SHA, "fixture/repo")[0]
    assert citation.valid and citation.evidence_id == details.id
    excerpt = json.loads(citation.cited_excerpt or "null")
    if fault is None:
        assert [r["id"] for r in excerpt["observations"]] == [details.id, diff.id]
        assert excerpt["observations"][1]["result"]["diff"] == "-before\n+after"
    else:
        assert "observations" not in excerpt
    details.result["pull_request"].pop("url")
    assert not referenced_citations(value, CORPUS, SHA, "fixture/repo")[0].valid


def test_pr_provenance_does_not_include_other_search_results() -> None:
    url = "https://github.com/fixture/repo/pull/17"
    value = result(f"The patch changes behavior ({url}).")
    value.tool_executions = [
        ToolExecution(
            id="tool:search",
            name="search_pull_requests",
            arguments={"query": "behavior"},
            result={
                "items": [
                    {"number": 17, "url": url, "body": "Requested proposal."},
                    {
                        "number": 18,
                        "url": "https://github.com/fixture/repo/pull/18",
                        "body": "UNRELATED_FACT",
                    },
                ]
            },
        ),
        ToolExecution(
            id="tool:diff",
            name="pull_request_read",
            arguments={"pull_number": 17, "method": "get_diff"},
            result={
                "repository": "fixture/repo",
                "pull_number": 17,
                "diff": "-before\n+after",
            },
        ),
    ]
    value.citation_catalog[url] = "tool:search"
    citation = referenced_citations(value, CORPUS, SHA, "fixture/repo")[0]
    assert citation.valid
    assert "Requested proposal." in (citation.cited_excerpt or "")
    assert "-before" in (citation.cited_excerpt or "")
    assert "UNRELATED_FACT" not in (citation.cited_excerpt or "")


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
            "get_repository_head",
            "search_issues",
            "issue_read",
            "search_pull_requests",
            "pull_request_read",
        }
        definitions = await execution.mcp.list_tools()
        assert {t.name: t.description for t in tools} == {
            t.name: t.description for t in definitions
        }
        issue_tool = next(t for t in tools if t.name == "issue_read")
        response = await issue_tool.ainvoke({"issue_number": 101})
        assert isinstance(response, dict) and "issue" in response
        await issue_tool.ainvoke({"issue_number": 999})
        assert records[0].arguments["method"] == "get"
        assert records[0].error is None
        assert records[1].fixture_miss and records[1].result == {}
        comments_fixture = next(
            call
            for call in fixtures.scenarios["streaming"]
            if call.arguments[0].get("method") == "get_comments"
        )
        original_comments = comments_fixture.model_dump_json()
        response = await issue_tool.ainvoke(
            {"issue_number": 101, "method": "get_comments"}
        )
        assert (
            response["url"] == "https://github.com/AdityaSaxena17/RepoMind/issues/101"
        )
        assert response["comments"] == comments_fixture.result["comments"]
        assert not records[-1].fixture_miss and records[-1].error is None
        assert comments_fixture.model_dump_json() == original_comments
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


def test_no_argument_fixture_tool_reports_missing_coverage_without_pagination() -> None:
    async def run() -> None:
        records: list[ToolExecution] = []
        tools = await fixture_tools(load_benchmark().fixtures, "streaming", records)
        tool = next(t for t in tools if t.name == "get_repository_head")
        await tool.ainvoke({})
        assert records[0].arguments == {}
        assert records[0].fixture_miss
        assert records[0].error == "Unmatched deterministic fixture call."

    asyncio.run(run())


def test_fixture_callables_are_accepted_by_actual_agent_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain.agents import create_agent

    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    monkeypatch.delenv("HIVE_API_KEY", raising=False)
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
    from app.config import get_settings

    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    get_settings.cache_clear()
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
    get_settings.cache_clear()


def test_embedding_cache_keeps_openai_and_qwen_vectors_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import get_settings

    calls: list[str] = []

    class Provider(FakeEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            settings = get_settings()
            calls.append(settings.embeddings_provider)
            return [[1.0] + [0.0] * (settings.embedding_dimensions - 1) for _ in texts]

    monkeypatch.setattr(execution, "create_embedding_model", Provider)
    try:
        for provider, dimensions in [
            ("openai", 1536),
            ("ollama", 1024),
            ("openai", 1536),
        ]:
            monkeypatch.setenv("EMBEDDINGS_PROVIDER", provider)
            get_settings.cache_clear()
            cached = CachedEmbeddings(tmp_path)
            assert len(asyncio.run(cached.aembed_query("same query"))) == dimensions
        assert calls == ["openai", "ollama"]
        assert len(list(tmp_path.glob("*.json"))) == 2
    finally:
        get_settings.cache_clear()


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


@pytest.mark.parametrize("model", ["offline-test", "deepseek-ai/deepseek-v4.1-flash"])
@pytest.mark.parametrize("bad_attempts", [0, 1, 3])
def test_judge_tasks_keep_gold_out_of_grounding_and_bound_validation_retries(
    monkeypatch: pytest.MonkeyPatch,
    bad_attempts: int,
    model: str,
) -> None:
    from openai.types.chat import ChatCompletion

    from app.evals import judge as judge_module

    monkeypatch.setenv("HIVE_API_KEY", "test-hive-key")
    payloads: list[str] = []
    value = result("alpha is 1.")
    q = question()
    q.reference_facts[0].text = "SECRET_GOLD_FACT_FOR_TEST"

    class FakeCall:
        def __init__(self, schema: type[object]) -> None:
            self.schema = schema

        async def ainvoke(self, messages: list[tuple[str, str]]) -> object:
            payloads.append(messages[-1][1])
            if self.schema in (GroundingBatch, judge_module._AnchoredGroundingBatch):
                labeled = GroundingBatch(
                    sections=[
                        GroundingSection(
                            section_id="section:1",
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
                            ],
                        )
                    ]
                )
                return judge_module._AnchoredGroundingBatch.model_validate(
                    labeled.model_dump()
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

    class FakeHiveClient:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["api_key"] == "test-hive-key"
            assert kwargs["base_url"] == "https://api.thehive.ai/api/v3"
            self.chat = self
            self.completions = self

        async def __aenter__(self) -> "FakeHiveClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def create(self, *, messages: list[dict[str, str]], **_kwargs: object):
            schema = (
                GroundingBatch
                if "GroundingBatch" in messages[0]["content"]
                else CorrectnessJudgment
            )
            labeled = await FakeCall(schema).ainvoke(
                [(message["role"], message["content"]) for message in messages]
            )
            return ChatCompletion.model_validate(
                {
                    "id": "offline",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": labeled.model_dump_json(),
                            },
                        }
                    ],
                }
            )

    monkeypatch.setattr(judge_module, "ChatOpenAI", create_chat)
    monkeypatch.setattr(judge_module, "AsyncOpenAI", FakeHiveClient)
    judge = judge_module.SemanticJudge(model, reasoning_effort="low")
    if model == "offline-test":
        assert configuration["temperature"] is None
        assert configuration["reasoning"] == {"effort": "low"}
    else:
        assert not configuration
        assert judge.configuration["provider"] == "hive"
        assert "test-hive-key" not in json.dumps(judge.configuration)
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
    "failure", [None, "malformed", "truncated", "empty", "choices", "auth", "retry"]
)
def test_hive_sdk_transport_validation_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    import httpx
    from openai import AsyncOpenAI, AuthenticationError

    from app.evals import judge as judge_module

    monkeypatch.setenv("HIVE_API_KEY", "test-hive-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-must-not-be-used")
    requests: list[dict[str, object]] = []
    clients: list[httpx.AsyncClient] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.thehive.ai/api/v3/chat/completions"
        assert request.headers["authorization"] == "Bearer test-hive-key"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-ai/deepseek-v4.1-flash"
        assert payload["reasoning_effort"] == "high"
        assert payload["max_completion_tokens"] == 16384
        assert payload["max_thinking_tokens"] == 8192
        assert payload["response_format"] == {"type": "json_object"}
        assert "JSON Schema" in payload["messages"][0]["content"]
        assert "temperature" not in payload
        requests.append(payload)
        if failure == "auth":
            return httpx.Response(401, json={"error": {"message": "offline"}})
        if failure == "retry" and len(requests) == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "offline"}},
                headers={"retry-after": "0"},
            )
        content = json.dumps({"claims": [], "limitations": []})
        if failure == "malformed":
            content = '{"claims": "wrong type"}'
        elif failure == "empty":
            content = ""
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": []
                if failure == "choices"
                else [
                    {
                        "index": 0,
                        "finish_reason": "length" if failure == "truncated" else "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    def create_client(**kwargs: object) -> AsyncOpenAI:
        transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(transport)
        return AsyncOpenAI(http_client=transport, **kwargs)

    monkeypatch.setattr(judge_module, "AsyncOpenAI", create_client)

    async def run() -> None:
        judge = judge_module.SemanticJudge("deepseek-ai/deepseek-v4.1-flash")
        messages = [("system", "Label claims."), ("user", "Answer.")]
        if failure == "auth":
            with pytest.raises(AuthenticationError):
                await judge._label(GroundingJudgment, messages)
        elif failure not in {None, "retry"}:
            with pytest.raises(ValueError):
                await judge._label(GroundingJudgment, messages)
        else:
            assert await judge._label(GroundingJudgment, messages) == GroundingJudgment(
                claims=[], limitations=[]
            )

    asyncio.run(run())
    assert len(requests) == (2 if failure == "retry" else 1)
    assert all(client.is_closed for client in clients)


def test_hive_key_is_required_without_openai_credential_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.evals.judge import SemanticJudge

    monkeypatch.delenv("HIVE_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-old-name-must-not-be-used")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-must-not-be-used")
    with pytest.raises(ValueError, match="HIVE_API_KEY"):
        SemanticJudge("deepseek-ai/deepseek-v4.1-flash")


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

    from app.config import get_settings
    from app.evals import __main__ as cli

    monkeypatch.setenv("HIVE_API_KEY", "test-hive-key")
    monkeypatch.setenv("HIVE_CHAT_MODEL", "deepseek-ai/deepseek-v4.1-flash")
    get_settings.cache_clear()

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
    monkeypatch.setattr(cli, "RetrievalRunner", FakeRunner)
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
    models = report.metadata["model_configurations"]
    assert models["chat"] == "deepseek-ai/deepseek-v4.1-flash"
    assert models["chat_provider"] == "hive"
    assert models["use_responses_api"] is False
    assert models["chat_reasoning_effort"] == "high"
    assert models["chat_max_thinking_tokens"] == 8192
    assert "test-hive-key" not in json.dumps(report.metadata)
    get_settings.cache_clear()


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
