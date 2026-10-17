"""Validated benchmark, runner, and judgment contracts independent of agents."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Span(Model):
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    file_sha256: str
    excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self) -> Span:
        """Reject inverted source locations at the dataset boundary."""
        if self.end_line < self.start_line:
            raise ValueError("Source span is inverted.")
        return self


class EvidenceUnit(Model):
    id: str
    alternatives: list[list[Span]] = Field(min_length=1)

    @model_validator(mode="after")
    def nonempty(self) -> EvidenceUnit:
        """Each alternative must contain actual evidence, never an empty proof."""
        if any(not spans for spans in self.alternatives):
            raise ValueError("Evidence alternatives cannot be empty.")
        return self


class ReferenceFact(Model):
    id: str
    text: str
    evidence_units: list[str] = Field(default_factory=list)
    fixture_evidence: list[str] = Field(default_factory=list)


class Question(Model):
    id: str
    category: Literal["single_file", "architecture", "insufficient", "github", "mixed"]
    evidence_scope: Literal["CODE", "GITHUB", "MIXED"]
    question: str
    required_evidence: list[EvidenceUnit] = Field(default_factory=list)
    relevant_evidence: list[EvidenceUnit] = Field(default_factory=list)
    reference_facts: list[ReferenceFact] = Field(default_factory=list)
    fixture_scenario: str = "none"
    expected_missing: list[str] = Field(default_factory=list)
    review_note: str


class FixtureCall(Model):
    id: str
    tool: str
    arguments: list[dict[str, JsonValue]] = Field(min_length=1)
    result: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None


class Fixtures(Model):
    version: str
    provenance: str
    scenarios: dict[str, list[FixtureCall]]


class Dataset(Model):
    version: str
    repository: str
    corpus_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    annotation_policy: str
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers(self) -> Dataset:
        """Require unique case/fact/unit IDs and resolvable fact annotations."""
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("Duplicate question ID.")
        for question in self.questions:
            units = question.required_evidence + question.relevant_evidence
            unit_ids = {unit.id for unit in units}
            if len(unit_ids) != len(units):
                raise ValueError(f"Duplicate evidence ID in {question.id}.")
            facts = question.reference_facts
            if len({fact.id for fact in facts}) != len(facts):
                raise ValueError(f"Duplicate fact ID in {question.id}.")
            if any(set(f.evidence_units) - unit_ids for f in facts):
                raise ValueError(f"Unknown evidence ID in {question.id}.")
        return self


class CodeContext(Model):
    id: str
    corpus_commit_sha: str
    source_index: int = Field(ge=1)
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str


class ToolExecution(Model):
    id: str
    name: str
    arguments: dict[str, JsonValue]
    result: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None
    fixture_miss: bool = False


class EvaluationResult(Model):
    answer: str = ""
    final_contexts: list[CodeContext] = Field(default_factory=list, max_length=6)
    evidence_seen: list[CodeContext] = Field(default_factory=list)
    citation_catalog: dict[str, str] = Field(default_factory=dict)
    tool_executions: list[ToolExecution] = Field(default_factory=list)


class Runner(Protocol):
    async def run(self, question: str) -> EvaluationResult:
        """Execute a question without receiving reference facts or gold spans."""
        ...


class Citation(Model):
    id: str
    text: str
    start: int
    end: int
    key: str
    evidence_id: str | None = None
    valid: bool
    reason: str
    cited_excerpt: str | None = None
    attached_range: tuple[int, int] | None = None

    @model_validator(mode="after")
    def ordered_attachment(self) -> Citation:
        if self.attached_range is not None and not (
            0 <= self.attached_range[0] < self.attached_range[1]
        ):
            raise ValueError("Citation attachment offsets must be ordered.")
        return self


class CitationSupport(Model):
    citation_id: str = Field(
        description="ID from referenced_citations, e.g. citation:1"
    )
    contributes: bool = Field(
        description="True only for evidence supporting a factual part of the claim; "
        "contradictory evidence and mere topical relevance do not contribute."
    )


class Claim(Model):
    id: str
    quote: str
    supported: bool
    evidence_ids: list[str] = Field(
        description="Minimal evidence IDs establishing the claim; [] if unsupported",
    )
    citations: list[CitationSupport]
    citations_fully_support: bool = Field(
        description="Attached valid citations jointly establish the entire claim",
    )
    reason: str


class GroundingJudgment(Model):
    claims: list[Claim] = Field(
        description="Domain factual claims only; exclude evidence limitations/refusals",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Exact quotations acknowledging missing evidence or inability to answer; "
            "these are not domain factual claims"
        ),
    )


class GroundingSection(GroundingJudgment):
    """Account for one bounded answer section, including nonfactual prose."""

    section_id: str
    nonfactual_reason: str | None = Field(default=None, min_length=1)


class GroundingBatch(Model):
    sections: list[GroundingSection]


class FactJudgment(Model):
    fact_id: str
    status: Literal["matched", "missing", "contradicted"]
    reason: str


class CorrectnessJudgment(Model):
    facts: list[FactJudgment]
    material_errors: list[str]
    abstention: Literal["none", "correct", "unnecessary", "incorrect"] = Field(
        description=(
            "Empty expected_missing: none/unnecessary; nonempty: correct/incorrect"
        ),
    )
    reason: str


class Metric(Model):
    value: float | None = None
    status: Literal["scored", "not_applicable", "unscored", "error"] = "unscored"
    numerator: int = 0
    denominator: int = 0


class CaseReport(Model):
    case_id: str
    category: str
    question: str
    status: Literal["completed", "error"] = "completed"
    error: str | None = None
    result: EvaluationResult = Field(default_factory=EvaluationResult)
    citations: list[Citation] = Field(default_factory=list)
    grounding: GroundingJudgment | None = None
    grounding_sections: list[GroundingSection] = Field(default_factory=list)
    correctness: CorrectnessJudgment | None = None
    judge_validation_errors: list[str] = Field(default_factory=list)
    metrics: dict[str, Metric] = Field(default_factory=dict)
    partial_overlap: dict[str, float] = Field(default_factory=dict)


class RunReport(Model):
    metadata: dict[str, JsonValue]
    cases: list[CaseReport] = Field(default_factory=list)
