"""Section coverage, retry cleanup and URL citation regressions for evaluator v2."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from test_evals import CORPUS, SHA, question, result

from app.evals.__main__ import SCORER_V1_SHA256, SCORER_VERSION, grade_report
from app.evals.execution import DATA, digest, load_benchmark
from app.evals.judge import SemanticJudge, grounding_sections
from app.evals.models import (
    CaseReport,
    Claim,
    CorrectnessJudgment,
    EvaluationResult,
    FactJudgment,
    GroundingBatch,
    GroundingSection,
    RunReport,
)
from app.evals.scoring import referenced_citations


@pytest.mark.parametrize("inline", [False, True])
def test_localhost_port_is_not_a_file_citation(inline: bool) -> None:
    address = "http://127.0.0.1:8137/api/chat"
    value = result(
        f"Call `{address}`; see app.py:1-2 [1]."
        if inline
        else f"Call {address}; see app.py:1-2 [1]."
    )
    citations = referenced_citations(value, CORPUS, SHA, "owner/repo")
    assert not any(c.key == "//127.0.0.1:8137" for c in citations)
    assert any(c.key == "app.py:1-2" and c.valid for c in citations)
    assert any(c.key == "1" and c.valid for c in citations)


def test_sections_preserve_every_nonwhitespace_character_and_exact_offsets() -> None:
    answer = "  Heading\n\n" + "alpha is 1. " * 120 + "\n\nBeta is 2.\n"
    sections = grounding_sections(answer)
    assert len(sections) > 2
    covered: set[int] = set()
    for section in sections:
        assert section["text"] == answer[section["start"] : section["end"]]
        assert 0 < len(section["text"]) <= 600
        covered.update(range(section["start"], section["end"]))
    assert all(
        i in covered for i, character in enumerate(answer) if not character.isspace()
    )


@pytest.mark.parametrize(
    "fault", [None, "omit", "foreign_quote", "empty", "cited_nonfactual"]
)
def test_grounding_accounts_for_all_sections_and_rejects_incomplete_repairs(
    monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    value = result("alpha is 1 [1].\n\nBeta is 2 [1].")
    q = question()
    q.reference_facts[0].text = "SECRET_REFERENCE_FOR_TEST"
    case = CaseReport(
        case_id=q.id, category=q.category, question=q.question, result=value
    )
    case.citations = referenced_citations(value, CORPUS, SHA, "owner/repo")
    attempts = 0
    inputs: list[str] = []

    async def label(self: Any, schema: Any, messages: Any) -> Any:
        nonlocal attempts
        inputs.append(messages[1][1])
        if schema is CorrectnessJudgment:
            return CorrectnessJudgment(
                facts=[FactJudgment(fact_id="f1", status="matched", reason="Test")],
                material_errors=[],
                abstention="none",
                reason="Test",
            )
        attempts += 1
        payload = json.loads(messages[1][1])
        assert "SECRET_REFERENCE_FOR_TEST" not in messages[1][1]
        sections = payload["sections"][:1] if fault == "omit" else payload["sections"]
        return GroundingBatch(
            sections=[
                GroundingSection(
                    section_id=s["id"],
                    claims=[]
                    if fault in {"empty", "cited_nonfactual"}
                    else [
                        Claim(
                            id="c1",
                            quote="foreign text"
                            if fault == "foreign_quote"
                            else s["text"],
                            supported=True,
                            evidence_ids=[value.evidence_seen[0].id],
                            citations=[],
                            citations_fully_support=False,
                            reason="Observed code",
                        )
                    ],
                    nonfactual_reason="Heading"
                    if fault == "cited_nonfactual"
                    else None,
                )
                for s in sections
            ]
        )

    monkeypatch.setattr(SemanticJudge, "_label", label)
    judge = SemanticJudge("offline-test")
    if fault is None:
        asyncio.run(judge.grade(case, q))
        assert case.grounding is not None and len(case.grounding.claims) == 2
        assert len({c.id for c in case.grounding.claims}) == 2
        assert [s.section_id for s in case.grounding_sections] == [
            "section:1",
            "section:2",
        ]
        assert "SECRET_REFERENCE_FOR_TEST" in inputs[-1]
    else:
        with pytest.raises(ValueError):
            asyncio.run(judge.grade(case, q))
        assert attempts == 3
        assert case.grounding is None and not case.grounding_sections
        assert len(case.judge_validation_errors) == 3


def test_failed_grounding_batch_cancels_and_awaits_other_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    value = result("\n\n".join(f"Section {i}: " + "alpha " * 75 for i in range(16)))
    q = question()
    case = CaseReport(
        case_id=q.id, category=q.category, question=q.question, result=value
    )
    started, closed = asyncio.Event(), asyncio.Event()

    async def label(self: Any, schema: Any, messages: Any) -> Any:
        payload = json.loads(messages[1][1])
        if payload["sections"][0]["id"] == "section:1":
            await started.wait()
            raise ValueError("Controlled failure")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(SemanticJudge, "_label", label)

    async def run() -> None:
        with pytest.raises(ValueError, match="Controlled failure"):
            await SemanticJudge("offline-test").grade(case, q)
        assert closed.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("fault", [None, "unknown_scorer", "changed_calibration"])
def test_versioned_rescore_and_exact_calibration_reuse(
    tmp_path: Path, fault: str | None
) -> None:
    benchmark = load_benchmark()
    q = benchmark.dataset.questions[0]

    class Judge:
        configuration = {
            "version": "2",
            "model": "offline",
            "calibration_example_count": 1,
        }

        async def calibrate(self) -> Any:
            pytest.fail(
                "Explicit exact-configuration reuse must not call calibration again."
            )

        async def grade(self, case: Any, question: Any) -> None:
            from app.evals.models import GroundingJudgment

            case.grounding = GroundingJudgment(claims=[])
            case.correctness = CorrectnessJudgment(
                facts=[], material_errors=[], abstention="none", reason="Offline"
            )

    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        RunReport(
            metadata={
                "judge_configuration": {
                    **Judge.configuration,
                    **(
                        {"model": "different"} if fault == "changed_calibration" else {}
                    ),
                },
                "judge_calibrated": True,
                "judge_calibration": [{"id": "offline", "passed": True}],
            }
        ).model_dump_json(),
        encoding="utf-8",
    )
    report = RunReport(
        metadata={
            "mode": "full",
            "dataset_case_count": 24,
            "scorer_version": "1",
            "scorer_sha256": "unknown"
            if fault == "unknown_scorer"
            else SCORER_V1_SHA256,
            "dataset_sha256": digest((DATA / "dataset.json").read_text("utf-8")),
            "fixture_sha256": digest((DATA / "github.json").read_text("utf-8")),
        },
        cases=[
            CaseReport(
                case_id=q.id,
                category=q.category,
                question=q.question,
                result=EvaluationResult(answer="Use `http://127.0.0.1:8137`."),
            )
        ],
    )

    async def run() -> None:
        await grade_report(
            report,
            Judge(),
            tmp_path / "rescored.json",
            DATA / "dataset.json",
            DATA / "github.json",
            rescore=True,
            calibration_from=calibration_path,
        )

    if fault:
        with pytest.raises(ValueError, match="scorer|Calibration"):
            asyncio.run(run())
    else:
        asyncio.run(run())
        assert report.metadata["scorer_version"] == SCORER_VERSION
        assert report.metadata["rescored_from"]["scorer_sha256"] == SCORER_V1_SHA256
        assert report.cases[0].citations == []
        assert report.metadata["judge_calibrated"] is True


def test_supplemental_set_has_verified_spans_and_separate_strict_fixtures() -> None:
    benchmark = load_benchmark(
        DATA / "supplemental-v1.json", DATA / "supplemental-github-v1.json"
    )
    assert len(benchmark.dataset.questions) == 7
    assert not {q.id for q in benchmark.dataset.questions} & {
        q.id for q in load_benchmark().dataset.questions
    }
    assert "independently authored holdout" in benchmark.dataset.annotation_policy
