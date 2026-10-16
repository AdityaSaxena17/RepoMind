import argparse
import asyncio
from pathlib import Path
from typing import Any

import pytest

from app.agents import AgentFindings
from app.config import get_settings
from app.evals import __main__ as cli
from app.evals.__main__ import aggregate
from app.evals.execution import DATA, ROOT, score_final_evidence_coverage
from app.evals.models import (
    CaseReport,
    CodeContext,
    EvaluationResult,
    EvidenceUnit,
    Question,
    RunReport,
    Span,
)
from app.evals.scoring import referenced_citations, score_retrieval
from app.orchestration import RouteDecision, WorkflowResult


def test_iterative_coverage_is_separate_from_unchanged_first_pass_at_six() -> None:
    sha = "a" * 40
    corpus = {"app.py": "first\nsecond"}
    contexts = [
        CodeContext(
            id=f"code:{line}",
            corpus_commit_sha=sha,
            source_index=line,
            path="app.py",
            start_line=line,
            end_line=line,
            content=text,
        )
        for line, text in [(1, "first"), (2, "second")]
    ]
    question = Question(
        id="iterative",
        category="architecture",
        evidence_scope="CODE",
        question="Explain both lines.",
        review_note="Coverage mechanics.",
        required_evidence=[
            EvidenceUnit(
                id=f"unit:{line}",
                alternatives=[
                    [
                        Span(
                            path="app.py",
                            start_line=line,
                            end_line=line,
                            file_sha256="hash",
                            excerpt=text,
                        )
                    ]
                ],
            )
            for line, text in [(1, "first"), (2, "second")]
        ],
    )
    result = EvaluationResult(
        answer="Both lines [1], [2].",
        final_contexts=contexts[:1],
        evidence_seen=contexts,
        citation_catalog={"1": "code:1", "2": "code:2"},
    )
    first, _ = score_retrieval(question, result, corpus, sha)
    coverage, overlaps = score_final_evidence_coverage(question, result, corpus, sha)
    assert first["recall_at_6"].value == 0.5
    assert first["precision_at_6"].value == 1
    assert coverage.value == 1
    assert overlaps == {"unit:1": 1, "unit:2": 1}
    citations = referenced_citations(result, corpus, sha, "fixture/repo")
    assert len(citations) == 2 and all(c.valid for c in citations)
    report = RunReport(
        metadata={},
        cases=[
            CaseReport(
                case_id="iterative",
                category="architecture",
                question=question.question,
                result=result,
                metrics={**first, "final_evidence_coverage": coverage},
            )
        ],
    )
    assert aggregate(report)["final_evidence_coverage"]["macro_average"] == 1
    report.cases[0].metrics.pop("final_evidence_coverage")
    assert "final_evidence_coverage" not in aggregate(report)
    result.evidence_seen[1].content = "fabricated"
    with pytest.raises(ValueError, match="frozen source"):
        score_final_evidence_coverage(question, result, corpus, sha)


@pytest.mark.parametrize("fail_first", [False, True])
def test_specialist_cli_records_findings_and_continues_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_first: bool,
) -> None:
    role = "graph"
    monkeypatch.setenv("HIVE_API_KEY", "offline")
    monkeypatch.setenv("SYNTHESIS_REASONING_EFFORT", "low")
    get_settings.cache_clear()
    prompts: list[str] = []
    closed: list[bool] = []

    class Connection:
        async def close(self) -> None:
            closed.append(True)

    async def connect() -> Connection:
        return Connection()

    async def verified(*args: Any) -> bool:
        return True

    class Runner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert len(args) == 8
            assert args[5].reasoning_effort == "high"
            assert kwargs["router_model"].reasoning_effort == "none"
            assert kwargs["synthesis_model"].reasoning_effort == "low"
            self.last_result = EvaluationResult()
            self.first_query = None
            self.statistics = {"completed_model_calls_including_router": 0}
            if role == "graph":
                self.last_findings = WorkflowResult(
                    decision=RouteDecision(
                        scope="MIXED",
                        clarification=None,
                        mixed_strategy="PARALLEL",
                        code_task="Inspect indexed code.",
                        github_task="Read live facts.",
                    ),
                    status="incomplete",
                    missing_information=["No supporting evidence was found."],
                )
            else:
                self.last_findings = AgentFindings(
                    role=role.upper(),
                    status="incomplete",
                    missing_information=["No supporting evidence was found."],
                )

        async def run(self, question: str) -> EvaluationResult:
            prompts.append(question)
            # An owning runner may finish deadline cleanup before returning findings.
            await asyncio.sleep(0.01)
            if fail_first and len(prompts) == 1:
                self.last_findings.status = "failed"
                raise RuntimeError("Controlled investigation failure.")
            return self.last_result

    monkeypatch.setattr(cli, "connect_evaluation_database", connect)
    monkeypatch.setattr(cli, "verify_index", verified)
    monkeypatch.setattr(cli, "corpus_documents", lambda benchmark: ("test", []))
    monkeypatch.setattr(cli, "GraphRunner", Runner)
    output = tmp_path / "specialists.json"
    args = argparse.Namespace(
        command="full",
        agent=role,
        tool_call_limit=2,
        model_call_limit=3,
        timeout=0.001,
        output=output,
        experiment="offline",
        dataset=DATA / "dataset.json",
        fixtures=DATA / "github.json",
        corpus_root=ROOT,
        case=["code-01", "github-01", "mixed-01"]
        if role == "graph"
        else [f"{role}-01", f"{role}-02"],
        rerank=False,
        judge_model=None,
    )
    try:
        assert asyncio.run(cli.run(args)) == int(fail_first)
        report = RunReport.model_validate_json(output.read_text("utf-8"))
        assert report.metadata["architecture"] == (
            "multi_agent_graph" if role == "graph" else f"{role}_agent"
        )
        assert report.metadata["agent_limits"]["tool_calls"] == 2
        assert [case.question for case in report.cases] == prompts
        assert [case.status for case in report.cases] == [
            "error" if fail_first else "completed",
            *["completed"] * (len(args.case) - 1),
        ]
        details = report.metadata["investigations"]
        assert details[args.case[0]]["status"] == (
            "failed" if fail_first else "incomplete"
        )
        assert details[args.case[-1]]["missing_information"]
        assert ("final_evidence_coverage" in report.cases[1].metrics) == (
            role in {"code", "graph"}
        )
        assert closed == [True]
    finally:
        get_settings.cache_clear()
