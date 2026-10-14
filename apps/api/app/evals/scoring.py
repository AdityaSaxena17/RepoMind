"""Deterministic metrics over visible evidence and validated semantic labels."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from urllib.parse import unquote, urlsplit

from pydantic import JsonValue

from ..chat import github_line_url
from .models import (
    Citation,
    CodeContext,
    CorrectnessJudgment,
    EvaluationResult,
    EvidenceUnit,
    GroundingJudgment,
    Metric,
    Question,
)

METRIC_NAMES = (
    "recall_at_6",
    "precision_at_6",
    "groundedness",
    "answer_correctness",
    "fact_completeness",
    "citation_precision",
    "citation_coverage",
    "citation_validity",
    "correct_abstention",
    "unnecessary_abstention",
)


def ratio(numerator: int, denominator: int) -> Metric:
    """Return a scored fraction or explicit N/A for an empty denominator."""
    return Metric(
        value=numerator / denominator if denominator else None,
        status="scored" if denominator else "not_applicable",
        numerator=numerator,
        denominator=denominator,
    )


def visible_lines(
    context: CodeContext,
    corpus: dict[str, str],
    commit_sha: str,
) -> set[tuple[str, int]]:
    """Verify actual returned text; declared but absent lines never earn credit."""
    if context.corpus_commit_sha != commit_sha or context.path not in corpus:
        raise ValueError("Context belongs to an unknown corpus or path.")
    if context.end_line < context.start_line:
        raise ValueError("Context range is inverted.")
    if not context.content:
        return set()
    source = corpus[context.path].splitlines()
    length = len(context.content.split("\n"))
    stop = context.start_line - 1 + length
    if stop > len(source) or stop > context.end_line:
        raise ValueError("Context text exceeds its source range.")
    if "\n".join(source[context.start_line - 1 : stop]) != context.content:
        raise ValueError("Context text does not match the frozen source.")
    return {(context.path, line) for line in range(context.start_line, stop + 1)}


def unit_overlap(unit: EvidenceUnit, lines: set[tuple[str, int]]) -> float:
    """Find the best coverage of an explicitly annotated evidence alternative."""
    best = 0.0
    for alternative in unit.alternatives:
        required = {
            (span.path, line)
            for span in alternative
            for line in range(span.start_line, span.end_line + 1)
        }
        best = max(best, len(required & lines) / len(required))
    return best


def score_retrieval(
    question: Question,
    result: EvaluationResult,
    corpus: dict[str, str],
    commit_sha: str,
) -> tuple[dict[str, Metric], dict[str, float]]:
    """Score final parent windows, retaining duplicate slots and overlap detail."""
    windows = [visible_lines(c, corpus, commit_sha) for c in result.final_contexts]
    if not question.required_evidence:
        return {
            "recall_at_6": ratio(0, 0),
            "precision_at_6": ratio(0, 0),
        }, {}
    union: set[tuple[str, int]] = set().union(*windows)
    overlaps = {
        unit.id: unit_overlap(unit, union) for unit in question.required_evidence
    }
    relevant = question.required_evidence + question.relevant_evidence
    credited: set[frozenset[tuple[str, int]]] = set()
    hits = 0
    for window in windows:
        key = frozenset(window)
        if key not in credited and any(
            unit_overlap(unit, window) == 1 for unit in relevant
        ):
            hits += 1
        credited.add(key)
    precision = (
        ratio(hits, len(windows))
        if windows
        else Metric(value=0, status="scored", numerator=0, denominator=0)
    )
    return {
        "recall_at_6": ratio(sum(v == 1 for v in overlaps.values()), len(overlaps)),
        "precision_at_6": precision,
    }, overlaps


def result_urls(value: JsonValue) -> set[str]:
    """Collect structured URL fields, excluding links embedded in body text."""
    urls: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"url", "html_url"} and isinstance(item, str):
                urls.add(item)
            elif isinstance(item, dict | list):
                urls.update(result_urls(item))
    elif isinstance(value, list):
        for item in value:
            urls.update(result_urls(item))
    return urls


def url_evidence(value: JsonValue, url: str) -> str | None:
    """Restrict a tool citation to its URL-bearing entity, not other search results."""
    children: Iterable[JsonValue]
    if isinstance(value, dict):
        if any(value.get(key) == url for key in ("url", "html_url")):
            return json.dumps(value, ensure_ascii=False)
        children = value.values()
    elif isinstance(value, list):
        children = iter(value)
    else:
        return None
    for child in children:
        if isinstance(child, dict | list):
            found = url_evidence(child, url)
            if found is not None:
                return found
    return None


def referenced_citations(
    result: EvaluationResult,
    corpus: dict[str, str],
    commit_sha: str,
    repository: str,
) -> list[Citation]:
    """Resolve references in answer prose, never unused source cards or code blocks."""
    answer = result.answer
    masked = list(answer)
    fences = re.compile(r"```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)")
    for match in fences.finditer(answer):
        masked[match.start() : match.end()] = " " * (match.end() - match.start())
    location_prose = "".join(masked)
    for match in re.finditer(r"`[^`\n]*`", location_prose):
        masked[match.start() : match.end()] = " " * (match.end() - match.start())
    prose = "".join(masked)
    contexts = {c.id: c for c in result.evidence_seen}
    visible = {key: visible_lines(c, corpus, commit_sha) for key, c in contexts.items()}
    urls = {
        url: (tool.id, url_evidence(tool.result, url))
        for tool in result.tool_executions
        if tool.error is None
        for url in result_urls(tool.result)
    }
    occurrences: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()", prose):
        occurrences.extend(
            (match.start(), match.end(), number.strip())
            for number in match.group(1).split(",")
        )
    for match in re.finditer(r"https?://[^\s<>()\[\]]+", prose):
        url = match.group().rstrip(".,;:")
        occurrences.append((match.start(), match.start() + len(url), url))
    for match in re.finditer(
        r"[\w@./\[\]-]+\.[\w]+:\d+(?:-\d+)?",
        location_prose,
    ):
        if not any(first <= match.start() < last for first, last, _ in occurrences):
            occurrences.append((match.start(), match.end(), match.group()))
    citations: list[Citation] = []
    for start, end, key in sorted(occurrences):
        evidence_id = result.citation_catalog.get(key)
        valid = False
        excerpt: str | None = None
        reason = "Reference does not resolve to evidence seen by the model."
        if key.isdigit():
            context = contexts.get(evidence_id or "")
            if context is not None:
                expected = {
                    (context.path, line)
                    for line in range(context.start_line, context.end_line + 1)
                }
                valid = bool(expected) and expected <= visible[context.id]
                reason = (
                    "Source index resolves to visible code."
                    if valid
                    else ("Source range includes text not present in the context.")
                )
        elif key in urls:
            evidence_id, excerpt = urls[key]
            valid, reason = True, "URL was returned by a tool."
        elif not key.startswith(("https://", "http://")):
            path, _, line_range = key.rpartition(":")
            first_text, _, last_text = line_range.partition("-")
            first, last = int(first_text), int(last_text or first_text)
            if path in corpus and 1 <= first <= last <= len(corpus[path].splitlines()):
                requested = {(path, line) for line in range(first, last + 1)}
                for context_id, lines in visible.items():
                    if requested <= lines:
                        evidence_id, valid = context_id, True
                        excerpt = "\n".join(corpus[path].splitlines()[first - 1 : last])
                        reason = "File location resolves to visible source lines."
                        break
        else:
            parsed = urlsplit(key)
            prefix = f"/{repository}/blob/{commit_sha}/"
            location = re.fullmatch(r"L(\d+)(?:-L(\d+))?", parsed.fragment)
            if (
                parsed.scheme == "https"
                and parsed.netloc == "github.com"
                and parsed.path.startswith(prefix)
                and not parsed.query
                and location
            ):
                path = unquote(parsed.path[len(prefix) :])
                first = int(location.group(1))
                last = int(location.group(2) or first)
                if path in corpus and 1 <= first <= last <= len(
                    corpus[path].splitlines()
                ):
                    requested = {(path, line) for line in range(first, last + 1)}
                    for context_id, lines in visible.items():
                        if requested <= lines:
                            evidence_id, valid = context_id, True
                            excerpt = "\n".join(
                                corpus[path].splitlines()[first - 1 : last]
                            )
                            reason = "Commit-pinned URL resolves to visible code."
                            break
        citations.append(
            Citation(
                id=f"citation:{len(citations) + 1}",
                text=answer[start:end],
                start=start,
                end=end,
                key=key,
                evidence_id=evidence_id,
                valid=valid,
                reason=reason,
                cited_excerpt=excerpt,
            )
        )
    return citations


def build_catalog(result: EvaluationResult, repository: str) -> dict[str, str]:
    """Create resolvable labels without treating their display as citation usage."""
    catalog = {str(c.source_index): c.id for c in result.final_contexts}
    for context in result.evidence_seen:
        url = github_line_url(
            f"https://github.com/{repository}",
            context.corpus_commit_sha,
            context.path,
            context.start_line,
            context.end_line,
        )
        if url:
            catalog[url] = context.id
    for tool in result.tool_executions:
        if tool.error is None:
            catalog.update(dict.fromkeys(result_urls(tool.result), tool.id))
    return catalog


def validate_grounding(
    judgment: GroundingJudgment,
    result: EvaluationResult,
    citations: list[Citation],
) -> None:
    """Reject invented quotations, evidence IDs, and impossible citation support."""
    evidence_ids = {c.id for c in result.evidence_seen} | {
        t.id for t in result.tool_executions if t.error is None
    }
    refs = {c.id: c for c in citations}
    if len({claim.id for claim in judgment.claims}) != len(judgment.claims):
        raise ValueError("Judge repeated a claim ID.")
    for claim in judgment.claims:
        if not claim.quote or claim.quote not in result.answer:
            raise ValueError("Judge quote does not occur in the answer.")
        if set(claim.evidence_ids) - evidence_ids:
            raise ValueError("Judge referenced evidence never seen by the model.")
        if claim.supported and not claim.evidence_ids:
            raise ValueError("Supported claims require evidence IDs.")
        if len({c.citation_id for c in claim.citations}) != len(claim.citations):
            raise ValueError("Judge repeated a claim-citation association.")
        for association in claim.citations:
            reference = refs.get(association.citation_id)
            if reference is None:
                raise ValueError("Judge invented a citation reference.")
            if association.contributes and not reference.valid:
                raise ValueError("Invalid reference marked supportive")
        if claim.citations_fully_support and (
            not claim.supported or not any(p.contributes for p in claim.citations)
        ):
            raise ValueError("Complete citation support lacks supporting references.")


def score_semantics(
    question: Question,
    result: EvaluationResult,
    citations: list[Citation],
    grounding: GroundingJudgment | None,
    correctness: CorrectnessJudgment | None,
) -> dict[str, Metric]:
    """Calculate separate metrics from categorical decisions, never model scores."""
    metrics = {name: Metric() for name in METRIC_NAMES[2:]}
    metrics["citation_validity"] = ratio(
        sum(c.valid for c in citations),
        len(citations),
    )
    if grounding is not None:
        validate_grounding(grounding, result, citations)
        claims = grounding.claims
        # The judge extracts domain factual claims; all of them require evidence.
        required = claims
        pairs = [pair for c in claims for pair in c.citations]
        associated = {pair.citation_id for pair in pairs}
        orphan_count = sum(c.id not in associated for c in citations)
        covered = sum(c.citations_fully_support for c in required)
        metrics.update(
            {
                "groundedness": ratio(sum(c.supported for c in claims), len(claims)),
                "citation_precision": ratio(
                    sum(p.contributes for p in pairs), len(pairs) + orphan_count
                ),
                "citation_coverage": ratio(
                    covered,
                    len(required),
                ),
            }
        )
    if correctness is not None:
        expected = {fact.id for fact in question.reference_facts}
        actual = [fact.fact_id for fact in correctness.facts]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("Judge must label every reference fact exactly once.")
        allowed_abstentions = (
            {"correct", "incorrect"}
            if question.expected_missing
            else {"none", "unnecessary"}
        )
        if correctness.abstention not in allowed_abstentions:
            raise ValueError("Abstention label disagrees with annotated limitations.")
        matched = sum(f.status == "matched" for f in correctness.facts)
        complete = matched == len(expected) and not correctness.material_errors
        metrics["fact_completeness"] = ratio(matched, len(expected))
        metrics["answer_correctness"] = (
            ratio(int(complete), 1) if expected else ratio(0, 0)
        )
        metrics["correct_abstention"] = (
            ratio(int(correctness.abstention == "correct" and complete), 1)
            if question.expected_missing
            else ratio(0, 0)
        )
        metrics["unnecessary_abstention"] = (
            ratio(int(correctness.abstention == "unnecessary"), 1)
            if not question.expected_missing
            else ratio(0, 0)
        )
    return metrics
