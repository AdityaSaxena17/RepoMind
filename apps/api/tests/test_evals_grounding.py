"""Section coverage, retry cleanup and URL citation regressions for evaluator v2."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from test_evals import CORPUS, SHA, question, result

from app.evals.__main__ import (
    SCORER_V1_SHA256,
    SCORER_V2_SHA256,
    SCORER_VERSION,
    grade_report,
)
from app.evals.execution import DATA, digest, load_benchmark
from app.evals.judge import SemanticJudge, grounding_sections
from app.evals.models import (
    CaseReport,
    CitationSupport,
    Claim,
    CorrectnessJudgment,
    EvaluationResult,
    FactJudgment,
    GroundingBatch,
    GroundingJudgment,
    GroundingSection,
    RunReport,
)
from app.evals.scoring import referenced_citations, score_semantics, validate_grounding


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


@pytest.mark.parametrize("language", ["python", "mermaid"])
@pytest.mark.parametrize(
    "caption", ["Sources: [1].", "Sources: alpha is 1 [1].", "[1]."]
)
def test_only_explicit_citation_only_captions_attach_to_fenced_blocks(
    language: str, caption: str
) -> None:
    answer = f"```{language}\nalpha = 1\n```\n\n{caption}\n\nBeta is 2."
    value = result(answer)
    citations = referenced_citations(value, CORPUS, SHA, "owner/repo")
    assert len(citations) == 1 and citations[0].valid
    citation = citations[0]
    assert answer[citation.start : citation.end] == "[1]"
    assert (citation.attached_range is not None) == (caption == "Sources: [1].")
    if citation.attached_range:
        first, last = citation.attached_range
        assert answer[first:last] == f"```{language}\nalpha = 1\n```"
        inside = Claim(
            id="inside",
            quote="alpha = 1",
            supported=True,
            evidence_ids=[value.evidence_seen[0].id],
            citations=[CitationSupport(citation_id=citation.id, contributes=True)],
            citations_fully_support=True,
            reason="Observed source and explicit caption.",
        )
        validate_grounding(GroundingJudgment(claims=[inside]), value, citations)
        outside = inside.model_copy(update={"quote": "Beta is 2."})
        with pytest.raises(ValueError, match="attached block"):
            validate_grounding(GroundingJudgment(claims=[outside]), value, citations)


def test_fenced_reference_text_and_distant_sources_do_not_create_attachments() -> None:
    answer = "```python\nalpha = 1  # [99]\n```\n\nUnrelated prose.\n\nSources: [1]."
    citations = referenced_citations(result(answer), CORPUS, SHA, "owner/repo")
    assert [c.key for c in citations] == ["1"]
    assert citations[0].attached_range is None


@pytest.mark.parametrize(
    "parts",
    [
        [
            "- First " + "condition " * 45 + "holds [1].",
            "- Second " + "effect " * 60 + "follows [1].",
        ],
        ["The " + "condition " * 75 + "uses `repo:sample/project is:issue` [1]."],
        ["The " + "condition " * 58 + "holds. [1][1].", "Another fact holds [1]."],
    ],
)
def test_sections_keep_bullets_long_clauses_and_inline_code_whole(
    parts: list[str],
) -> None:
    answer = "\n".join(parts)
    sections = grounding_sections(answer)
    assert all(any(part in section["text"] for section in sections) for part in parts)
    assert all(
        section["text"] == answer[section["start"] : section["end"]]
        for section in sections
    )
    assert all(len(section["text"]) <= 2400 for section in sections)


def test_sections_bound_unbroken_input_without_losing_characters() -> None:
    answer = "x" * 5001
    sections = grounding_sections(answer)
    assert "".join(section["text"] for section in sections) == answer
    assert all(0 < len(section["text"]) <= 2400 for section in sections)


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
        if fault != "omit":
            assert case.judge_validation_errors[0].startswith("section:1:")


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


@pytest.mark.parametrize(
    "references",
    [
        "[1][1].",
        "([1], [1]);",
        "Sources: [1].</response>\n</wrapper>",
        "Sources: [1].</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    ],
)
def test_citation_only_sections_are_formatting_without_dropping_claims(
    monkeypatch: pytest.MonkeyPatch, references: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    value = result("alpha is 1 [1].\n\n" + references)
    q = question()
    case = CaseReport(
        case_id=q.id, category=q.category, question=q.question, result=value
    )
    case.citations = referenced_citations(value, CORPUS, SHA, "owner/repo")

    async def label(self: Any, schema: Any, messages: Any) -> Any:
        if schema is CorrectnessJudgment:
            return CorrectnessJudgment(
                facts=[FactJudgment(fact_id="f1", status="matched", reason="Test")],
                material_errors=[],
                abstention="none",
                reason="Test",
            )
        sections = json.loads(messages[1][1])["sections"]
        return GroundingBatch(
            sections=[
                GroundingSection(
                    section_id=s["id"],
                    claims=[
                        Claim(
                            id="c1",
                            quote=s["text"],
                            supported=True,
                            evidence_ids=[value.evidence_seen[0].id],
                            citations=[],
                            citations_fully_support=False,
                            reason="Observed code",
                        )
                    ]
                    if i == 0
                    else [],
                    nonfactual_reason=None if i == 0 else "Citation-only formatting",
                )
                for i, s in enumerate(sections)
            ]
        )

    monkeypatch.setattr(SemanticJudge, "_label", label)
    asyncio.run(SemanticJudge("offline-test").grade(case, q))
    assert case.grounding is not None and len(case.grounding.claims) == 1
    assert len(case.grounding_sections) == 2
    assert not case.judge_validation_errors


@pytest.mark.parametrize(
    "text", ["VALUE is 99 [1].</response>", "The literal `</response>` [1]."]
)
def test_closing_markup_never_excuses_cited_factual_or_literal_prose(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    value = result(text)
    q = question()
    case = CaseReport(
        case_id=q.id, category=q.category, question=q.question, result=value
    )
    case.citations = referenced_citations(value, CORPUS, SHA, "owner/repo")

    async def label(self: Any, schema: Any, messages: Any) -> GroundingBatch:
        return GroundingBatch(
            sections=[
                GroundingSection(
                    section_id=s["id"], claims=[], nonfactual_reason="Formatting"
                )
                for s in json.loads(messages[1][1])["sections"]
            ]
        )

    monkeypatch.setattr(SemanticJudge, "_label", label)
    with pytest.raises(ValueError, match="Cited prose"):
        asyncio.run(SemanticJudge("offline-test").grade(case, q))
    assert case.grounding is None and len(case.judge_validation_errors) == 3


def test_grounding_batches_bound_output_and_quote_only_assigned_answer_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    value = result("\n\n".join(f"Statement {i}: " + "alpha " * 70 for i in range(9)))
    q = question()
    case = CaseReport(
        case_id=q.id, category=q.category, question=q.question, result=value
    )
    batches: list[list[dict[str, Any]]] = []

    async def label(self: Any, schema: Any, messages: Any) -> Any:
        payload = json.loads(messages[1][1])
        if schema is CorrectnessJudgment:
            assert payload["answer"] == value.answer
            return CorrectnessJudgment(
                facts=[FactJudgment(fact_id="f1", status="matched", reason="Test")],
                material_errors=[],
                abstention="none",
                reason="Test",
            )
        assert "answer" not in payload
        assert payload["evidence_seen"] == [c.model_dump() for c in value.evidence_seen]
        batch = payload["sections"]
        assert len(batch) <= 6
        assert sum(len(s["text"]) for s in batch) <= 1200
        batches.append(batch)
        return GroundingBatch(
            sections=[
                GroundingSection(
                    section_id=s["id"],
                    claims=[
                        Claim(
                            id="c1",
                            quote=s["text"],
                            supported=True,
                            evidence_ids=[value.evidence_seen[0].id],
                            citations=[],
                            citations_fully_support=False,
                            reason="Observed evidence",
                        )
                    ],
                )
                for s in batch
            ]
        )

    monkeypatch.setattr(SemanticJudge, "_label", label)
    asyncio.run(SemanticJudge("offline-test").grade(case, q))
    assert len(batches) >= 3
    assert case.grounding is not None and len(case.grounding.claims) == 9
    assert len(case.grounding_sections) == 9


def test_orphan_caption_reference_still_reduces_citation_precision() -> None:
    value = result("alpha is 1 [1].\n\nSources: [1].</response>")
    citations = referenced_citations(value, CORPUS, SHA, "owner/repo")
    assert len(citations) == 2 and all(c.valid for c in citations)
    grounding = GroundingJudgment(
        claims=[
            Claim(
                id="c1",
                quote="alpha is 1",
                supported=True,
                evidence_ids=[value.evidence_seen[0].id],
                citations=[
                    CitationSupport(citation_id=citations[0].id, contributes=True)
                ],
                citations_fully_support=True,
                reason="Observed value",
            )
        ]
    )
    metrics = score_semantics(question(), value, citations, grounding, None)
    assert metrics["citation_precision"].value == 0.5
    assert metrics["citation_coverage"].value == 1


@pytest.mark.parametrize("fault", [None, "unknown", "other_section", "contradictory"])
def test_quote_anchors_copy_exact_answer_lines_and_reject_forged_locations(
    monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    value = result("```mermaid\nflowchart LR\n    A[alpha is 1]\n```\n\nSources: [1].")
    q = question()
    case = CaseReport(
        case_id=q.id, category=q.category, question=q.question, result=value
    )
    case.citations = referenced_citations(value, CORPUS, SHA, "owner/repo")

    async def label(self: Any, schema: Any, messages: Any) -> Any:
        if schema is CorrectnessJudgment:
            return CorrectnessJudgment(
                facts=[FactJudgment(fact_id="f1", status="matched", reason="Test")],
                material_errors=[],
                abstention="none",
                reason="Test",
            )
        sections = json.loads(messages[1][1])["sections"]
        first, caption = sections
        anchor = next(
            key for key, text in first["quote_anchors"].items() if "alpha is 1" in text
        )
        if fault == "unknown":
            anchor = "invented"
        elif fault == "other_section":
            anchor = next(iter(caption["quote_anchors"]))
        return schema.model_validate(
            {
                "sections": [
                    {
                        "section_id": first["id"],
                        "claims": [
                            {
                                "id": "c1",
                                "quote_anchor": anchor,
                                "quote": "alpha is 1"
                                if fault == "contradictory"
                                else "",
                                "supported": True,
                                "evidence_ids": [value.evidence_seen[0].id],
                                "citations": [
                                    {
                                        "citation_id": case.citations[0].id,
                                        "contributes": True,
                                    }
                                ],
                                "citations_fully_support": True,
                                "reason": "Observed value",
                            }
                        ],
                    },
                    {
                        "section_id": caption["id"],
                        "claims": [],
                        "nonfactual_reason": "Citation-only formatting",
                    },
                ]
            }
        )

    monkeypatch.setattr(SemanticJudge, "_label", label)
    if fault is None:
        asyncio.run(SemanticJudge("offline-test").grade(case, q))
        assert case.grounding is not None
        claim = case.grounding.claims[0]
        assert claim.quote == "    A[alpha is 1]"
        assert "quote_anchor" not in claim.model_dump()
        assert claim.evidence_ids == [value.evidence_seen[0].id]
        assert case.metrics["citation_coverage"].value == 1
    else:
        with pytest.raises(ValueError, match="anchor"):
            asyncio.run(SemanticJudge("offline-test").grade(case, q))
        assert case.grounding is None and len(case.judge_validation_errors) == 3


@pytest.mark.parametrize("bad_attempts", [0, 1, 3])
def test_correctness_repairs_missing_fact_labels_without_rerunning_grounding(
    monkeypatch: pytest.MonkeyPatch, bad_attempts: int
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    value = result("alpha is 1.")
    q = question()
    case = CaseReport(
        case_id=q.id, category=q.category, question=q.question, result=value
    )
    calls = {"grounding": 0, "correctness": 0}

    async def label(self: Any, schema: Any, messages: Any) -> Any:
        if schema is CorrectnessJudgment:
            calls["correctness"] += 1
            return CorrectnessJudgment(
                facts=[]
                if calls["correctness"] <= bad_attempts
                else [
                    FactJudgment(
                        fact_id="f1", status="missing", reason="Valid low grade"
                    )
                ],
                material_errors=[],
                abstention="none",
                reason="Test",
            )
        calls["grounding"] += 1
        return GroundingBatch(
            sections=[
                GroundingSection(
                    section_id="section:1",
                    claims=[
                        Claim(
                            id="c1",
                            quote=value.answer,
                            supported=True,
                            evidence_ids=[value.evidence_seen[0].id],
                            citations=[],
                            citations_fully_support=False,
                            reason="Observed fact",
                        )
                    ],
                )
            ]
        )

    monkeypatch.setattr(SemanticJudge, "_label", label)
    if bad_attempts == 3:
        with pytest.raises(ValueError, match="every reference fact"):
            asyncio.run(SemanticJudge("offline-test").grade(case, q))
        assert case.grounding is None and case.correctness is None
    else:
        asyncio.run(SemanticJudge("offline-test").grade(case, q))
        assert case.metrics["fact_completeness"].value == 0
    assert calls["grounding"] == 1
    assert calls["correctness"] == min(bad_attempts + 1, 3)
    assert len(case.judge_validation_errors) == bad_attempts
    assert all(e.startswith("correctness:") for e in case.judge_validation_errors)


@pytest.mark.parametrize("fault", [None, "unknown_scorer", "changed_calibration"])
@pytest.mark.parametrize("old_version", ["1", "2"])
def test_versioned_rescore_and_exact_calibration_reuse(
    tmp_path: Path, fault: str | None, old_version: str
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
            "scorer_version": old_version,
            "scorer_sha256": "unknown"
            if fault == "unknown_scorer"
            else SCORER_V1_SHA256
            if old_version == "1"
            else SCORER_V2_SHA256,
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
        assert report.metadata["rescored_from"]["scorer_sha256"] == (
            SCORER_V1_SHA256 if old_version == "1" else SCORER_V2_SHA256
        )
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
