"""CLI for corpus preparation, quality runs, saved-answer grading, and comparison."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import inspect
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from .. import retrieval
from ..chat import CHAT_SYSTEM_PROMPT, agent_messages
from ..config import get_settings
from ..reranking import (
    RERANKER_BATCH_SIZE,
    RERANKER_INSTRUCTION,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_REVISION,
    get_reranker,
)
from ..worker import CHUNK_OVERLAP, CHUNK_TOKENS
from .execution import (
    ARTIFACTS,
    DATA,
    ROOT,
    CachedEmbeddings,
    SingleAgentRunner,
    connect_evaluation_database,
    corpus_documents,
    digest,
    fixture_tools,
    load_benchmark,
    prepare_index,
    verify_index,
)
from .judge import SemanticJudge
from .models import CaseReport, Metric, Runner, RunReport, ToolExecution
from .scoring import (
    METRIC_NAMES,
    referenced_citations,
    score_retrieval,
    score_semantics,
)

SCORER_VERSION = "1"


def aggregate(report: RunReport, category: str | None = None) -> dict[str, JsonValue]:
    """Macro-average metrics with applicability and missing-score counts."""
    cases = [c for c in report.cases if category is None or c.category == category]
    output: dict[str, JsonValue] = {}
    for name in METRIC_NAMES:
        metrics = [c.metrics.get(name, Metric()) for c in cases]
        values = [
            m.value for m in metrics if m.status == "scored" and m.value is not None
        ]
        output[name] = {
            "macro_average": sum(values) / len(values) if values else None,
            "scored_cases": len(values),
            "not_applicable": sum(m.status == "not_applicable" for m in metrics),
            "unscored": sum(m.status == "unscored" for m in metrics),
            "errors": sum(m.status == "error" for m in metrics),
            "numerator_sum": sum(m.numerator for m in metrics if m.status == "scored"),
            "denominator_sum": sum(
                m.denominator for m in metrics if m.status == "scored"
            ),
        }
    return output


def write_report(report: RunReport, path: Path) -> None:
    """Checkpoint JSON/Markdown with separate scores and no credentials."""
    report.metadata["aggregates"] = aggregate(report)
    report.metadata["by_category"] = {
        category: aggregate(report, category)
        for category in sorted({c.category for c in report.cases})
    }
    report.metadata["completed_cases"] = sum(
        c.status == "completed" for c in report.cases
    )
    report.metadata["failed_cases"] = sum(c.status == "error" for c in report.cases)
    complete = (
        len(report.cases) == report.metadata.get("dataset_case_count")
        and len({c.case_id for c in report.cases}) == len(report.cases)
        and all(c.status == "completed" for c in report.cases)
        and report.metadata.get("mode") == "full"
        and report.metadata.get("judge_calibrated") is True
        and all(
            c.grounding is not None and c.correctness is not None for c in report.cases
        )
    )
    report.metadata["complete_quality_baseline"] = complete
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    lines = [
        f"# Evaluation: {report.metadata.get('experiment_name')}",
        "",
        f"Complete quality baseline: **{complete}**. "
        f"Completed: {report.metadata['completed_cases']}; "
        f"failed: {report.metadata['failed_cases']}.",
        "",
        "GitHub results belong to the versioned synthetic fixture world.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(
            {
                k: v
                for k, v in report.metadata.items()
                if k not in {"aggregates", "by_category"}
            },
            indent=2,
        ),
        "```",
        "",
    ]
    for category in [None, *sorted({c.category for c in report.cases})]:
        lines.extend(
            [
                f"## {category or 'All cases'}",
                "",
                "| Metric | Macro average | Scored | N/A | Unscored | Errors |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, summary in aggregate(report, category).items():
            if isinstance(summary, dict):
                value = summary["macro_average"]
                rendered = f"{value:.4f}" if isinstance(value, float) else "â€”"
                lines.append(
                    f"| {name} | {rendered} | {summary['scored_cases']} | "
                    f"{summary['not_applicable']} | {summary['unscored']} | "
                    f"{summary['errors']} |",
                )
        lines.append("")
    for case in report.cases:
        lines.extend(
            [
                f"## {case.case_id}: {case.status}",
                "",
                case.question,
                "",
                case.result.answer
                or "No generated answer (retrieval-only or failed run).",
                "",
                f"Error: {case.error or 'none'}",
                "",
                "Final contexts:",
                "",
            ]
        )
        lines.extend(
            f"- [{c.source_index}] {c.path}:{c.start_line}-{c.end_line}"
            for c in case.result.final_contexts
        )
        lines.extend(
            [
                "",
                "Details:",
                "",
                "```json",
                json.dumps(
                    {
                        "metrics": {k: v.model_dump() for k, v in case.metrics.items()},
                        "partial_overlap": case.partial_overlap,
                        "citations": [c.model_dump() for c in case.citations],
                        "grounding": case.grounding.model_dump()
                        if case.grounding
                        else None,
                        "correctness": case.correctness.model_dump()
                        if case.correctness
                        else None,
                        "tool_executions": [
                            t.model_dump() for t in case.result.tool_executions
                        ],
                        "judge_validation_errors": case.judge_validation_errors,
                    },
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def compare_reports(before: RunReport, after: RunReport) -> dict[str, JsonValue]:
    """Compare paired cases only when gold, corpus, fixtures, and scoring match."""
    keys = (
        "dataset_version",
        "dataset_sha256",
        "fixture_version",
        "fixture_sha256",
        "corpus_commit_sha",
        "scorer_version",
        "scorer_sha256",
        "judge_configuration",
    )
    for key in keys:
        if key not in before.metadata or key not in after.metadata:
            raise ValueError(f"Reports are missing compatible {key}.")
        if before.metadata.get(key) != after.metadata.get(key):
            raise ValueError(f"Reports have incompatible {key}.")
    left = {c.case_id: c for c in before.cases}
    right = {c.case_id: c for c in after.cases}
    if (
        set(left) != set(right)
        or len(left) != len(before.cases)
        or len(right) != len(after.cases)
    ):
        raise ValueError("Comparison requires identical, unique case IDs.")
    paired: dict[str, JsonValue] = {}
    for name in METRIC_NAMES:
        deltas: dict[str, JsonValue] = {}
        for case_id in sorted(left):
            first = left[case_id].metrics.get(name, Metric())
            second = right[case_id].metrics.get(name, Metric())
            if (
                first.status == second.status == "scored"
                and first.value is not None
                and second.value is not None
            ):
                deltas[case_id] = second.value - first.value
        values = [v for v in deltas.values() if isinstance(v, float)]
        paired[name] = {
            "paired_cases": len(deltas),
            "per_case_delta": deltas,
            "macro_delta": sum(values) / len(values) if values else None,
        }
    return {
        "before_configuration": before.metadata,
        "after_configuration": after.metadata,
        "metrics": paired,
        "note": "Paired descriptive deltas; no statistical significance is claimed.",
    }


def judge_agreement(before: RunReport, after: RunReport) -> dict[str, JsonValue]:
    """Compare judges on identical saved evidence and answers, not RAG improvements."""
    for key in (
        "dataset_sha256",
        "fixture_sha256",
        "corpus_commit_sha",
        "scorer_sha256",
        "prompt_sha256",
        "model_configurations",
        "retrieval_parameters",
    ):
        if key not in before.metadata or before.metadata[key] != after.metadata.get(
            key
        ):
            raise ValueError(f"Judge trial requires matching {key}.")
    old_config, new_config = (
        before.metadata.get("judge_configuration"),
        after.metadata.get("judge_configuration"),
    )
    if not isinstance(old_config, dict) or not isinstance(new_config, dict):
        raise ValueError("Both reports need judge configurations.")
    for key in (
        "rubric_sha256",
        "implementation_sha256",
        "schema_sha256",
        "calibration_sha256",
    ):
        if key not in old_config or old_config[key] != new_config.get(key):
            raise ValueError(f"Judge trial requires matching {key}.")
    left = {c.case_id: c for c in before.cases}
    right = {c.case_id: c for c in after.cases}
    if (
        set(left) != set(right)
        or len(left) != len(before.cases)
        or len(right) != len(after.cases)
    ):
        raise ValueError("Judge trial requires identical unique case IDs.")
    fact_count = fact_matches = 0
    fact_differences: list[JsonValue] = []
    correctness_differences: list[JsonValue] = []
    abstention_differences: list[JsonValue] = []
    for case_id, old in left.items():
        new = right[case_id]
        if (
            old.question != new.question
            or old.result != new.result
            or old.citations != new.citations
        ):
            raise ValueError(
                "Judge trial requires identical saved answers and evidence."
            )
        for name in ("recall_at_6", "precision_at_6", "citation_validity"):
            if old.metrics.get(name) != new.metrics.get(name):
                raise ValueError("Deterministic scores changed in a judge-only trial.")
        old_facts = (
            {f.fact_id: f for f in old.correctness.facts} if old.correctness else {}
        )
        new_facts = (
            {f.fact_id: f for f in new.correctness.facts} if new.correctness else {}
        )
        if set(new_facts) - set(old_facts):
            raise ValueError("Candidate judge introduced unknown fact IDs.")
        for fact_id, fact in old_facts.items():
            candidate_fact = new_facts.get(fact_id)
            fact_count += 1
            if candidate_fact is not None and candidate_fact.status == fact.status:
                fact_matches += 1
            else:
                fact_differences.append(
                    {
                        "case_id": case_id,
                        "fact_id": fact_id,
                        "before": fact.status,
                        "after": candidate_fact.status if candidate_fact else None,
                        "before_reason": fact.reason,
                        "after_reason": candidate_fact.reason
                        if candidate_fact
                        else None,
                    }
                )
        old_metric = old.metrics.get("answer_correctness", Metric())
        new_metric = new.metrics.get("answer_correctness", Metric())
        if (old_metric.status, old_metric.value) != (
            new_metric.status,
            new_metric.value,
        ):
            correctness_differences.append(case_id)
        old_abstention = old.correctness.abstention if old.correctness else None
        new_abstention = new.correctness.abstention if new.correctness else None
        if old_abstention != new_abstention:
            abstention_differences.append(
                {
                    "case_id": case_id,
                    "before": old_abstention,
                    "after": new_abstention,
                }
            )
    semantic: dict[str, JsonValue] = {}
    semantic_agrees = True
    for name in ("groundedness", "citation_precision", "citation_coverage"):
        old_values: list[float] = []
        new_values: list[float] = []
        applicability: list[JsonValue] = []
        deltas: dict[str, JsonValue] = {}
        for case_id, old in left.items():
            a, b = (
                old.metrics.get(name, Metric()),
                right[case_id].metrics.get(name, Metric()),
            )
            if a.status != b.status:
                applicability.append(case_id)
            if a.status == "scored" and a.value is not None:
                old_values.append(a.value)
            if b.status == "scored" and b.value is not None:
                new_values.append(b.value)
            if (
                a.status == b.status == "scored"
                and a.value is not None
                and b.value is not None
            ):
                deltas[case_id] = b.value - a.value
        a_mean = sum(old_values) / len(old_values) if old_values else None
        b_mean = sum(new_values) / len(new_values) if new_values else None
        difference = (
            abs(b_mean - a_mean) if a_mean is not None and b_mean is not None else None
        )
        agrees = not applicability and (
            difference is not None
            and difference <= 0.05 + 1e-12
            or a_mean is b_mean is None
        )
        semantic_agrees &= agrees
        semantic[name] = {
            "before_macro": a_mean,
            "after_macro": b_mean,
            "absolute_macro_difference": difference,
            "passes": agrees,
            "applicability_disagreements": applicability,
            "per_case_delta": deltas,
        }
    calibration = after.metadata.get("judge_calibration")
    calibration_passes = (
        isinstance(calibration, list)
        and len(calibration) == 12
        and all(
            isinstance(row, dict) and row.get("passed") is True for row in calibration
        )
    )
    agreement = fact_matches / fact_count if fact_count else None
    gates: dict[str, JsonValue] = {
        "complete_runs": before.metadata.get("complete_quality_baseline") is True
        and after.metadata.get("complete_quality_baseline") is True
        and all(
            c.grounding is not None
            and c.correctness is not None
            and c.status == "completed"
            for c in [*left.values(), *right.values()]
        ),
        "calibration": calibration_passes,
        "correctness_outcomes": not correctness_differences,
        "abstention_outcomes": not abstention_differences,
        "fact_labels_at_least_95_percent": agreement is not None and agreement >= 0.95,
        "semantic_macros_within_5_points": semantic_agrees,
    }
    return {
        "before_judge": old_config,
        "candidate_judge": new_config,
        "gates": gates,
        "eligible_for_adoption": all(gates.values()),
        "requires_manual_review": True,
        "fact_label_agreement": agreement,
        "matched_fact_labels": fact_matches,
        "total_fact_labels": fact_count,
        "fact_disagreements": fact_differences,
        "correctness_disagreements": correctness_differences,
        "abstention_disagreements": abstention_differences,
        "semantic_metrics": semantic,
        "note": "Judge agreement on fixed answers; differences are not quality gains.",
    }


async def grade_report(
    report: RunReport,
    judge: SemanticJudge,
    path: Path,
    dataset_path: Path,
    fixtures_path: Path,
    corpus_root: Path = ROOT,
) -> None:
    """Calibrate and grade saved answers, retaining checkpoints after each case."""
    benchmark = load_benchmark(dataset_path, fixtures_path, corpus_root)
    if (
        report.metadata.get("dataset_sha256") != digest(dataset_path.read_text("utf-8"))
        or report.metadata.get("fixture_sha256")
        != digest(fixtures_path.read_text("utf-8"))
        or report.metadata.get("scorer_sha256")
        != digest(
            (Path(__file__).parent / "scoring.py").read_text("utf-8"),
        )
        or report.metadata.get("mode") != "full"
    ):
        raise ValueError("Saved run is not a compatible full-run artifact.")
    report.metadata["judge_configuration"] = judge.configuration
    report.metadata["judge_calibrated"] = False
    for case in report.cases:
        case.grounding, case.correctness = None, None
        for name in METRIC_NAMES[2:]:
            if name != "citation_validity":
                case.metrics[name] = Metric()
        if case.error and case.error.startswith("grading:"):
            case.status, case.error = "completed", None
    write_report(report, path)
    calibration = await judge.calibrate()
    report.metadata["judge_calibration"] = TypeAdapter(JsonValue).validate_python(
        calibration,
    )
    if not all(row["passed"] is True for row in calibration):
        write_report(report, path)
        raise ValueError("Judge failed calibration; semantic grades were not accepted.")
    report.metadata["judge_calibrated"] = True
    questions = {q.id: q for q in benchmark.dataset.questions}
    report.metadata["grading_case_concurrency"] = 3
    semaphore = asyncio.Semaphore(3)

    async def grade_case(case: CaseReport) -> None:
        async with semaphore:
            try:
                question = questions[case.case_id]
                if question.question != case.question:
                    raise ValueError("Saved question does not match the dataset.")
                case.citations = referenced_citations(
                    case.result,
                    benchmark.corpus,
                    benchmark.dataset.corpus_commit_sha,
                    benchmark.dataset.repository,
                )
                await judge.grade(case, question)
            except Exception as error:
                case.status, case.error = "error", f"grading: {type(error).__name__}"
                if type(error) is ValueError:
                    case.error += f": {error}"
                for metric in case.metrics.values():
                    if metric.status == "unscored":
                        metric.status = "error"
            write_report(report, path)
            print(f"Graded {case.case_id}: {case.status}", flush=True)

    await asyncio.gather(
        *(grade_case(case) for case in report.cases if case.status == "completed")
    )


async def run(args: argparse.Namespace) -> int:
    """Execute the selected explicit evaluation action, never on application startup."""
    if args.command == "migrate":
        connection = await connect_evaluation_database()
        await connection.close()
        environment = os.environ.copy()
        environment["DATABASE_URL"] = environment["EVAL_DATABASE_URL"]
        process = await asyncio.to_thread(
            subprocess.run,
            [
                "pnpm.cmd" if os.name == "nt" else "pnpm",
                "--filter",
                "@repo/db",
                "exec",
                "prisma",
                "migrate",
                "deploy",
                "--schema=./prisma/schema.prisma",
            ],
            cwd=ROOT,
            env=environment,
            timeout=120,
            check=False,
        )
        return process.returncode
    if args.command == "calibrate":
        if args.timeout <= 0:
            raise ValueError("Timeout must be positive.")
        judge = SemanticJudge(
            args.judge_model,
            timeout=args.timeout,
            reasoning_effort=args.reasoning_effort,
        )
        outcomes = await judge.calibrate()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "experiment_name": "judge-calibration-v1",
                    "configuration": judge.configuration,
                    "results": outcomes,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return int(not all(row["passed"] is True for row in outcomes))
    if args.command == "compare":
        before = RunReport.model_validate_json(args.before.read_text("utf-8"))
        after = RunReport.model_validate_json(args.after.read_text("utf-8"))
        comparison = compare_reports(before, after)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Comparison written to {args.output}")
        return 0
    if args.command == "judge-agreement":
        before = RunReport.model_validate_json(args.before.read_text("utf-8"))
        after = RunReport.model_validate_json(args.after.read_text("utf-8"))
        agreement = judge_agreement(before, after)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(agreement, indent=2) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(
            "# Judge agreement\n\nThis compares judges on identical saved answers. "
            "Manual review is required before adoption.\n\n```json\n"
            + json.dumps(agreement, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
        print(f"Judge agreement written to {args.output}")
        return 0
    if args.timeout <= 0:
        raise ValueError("Timeout must be positive.")
    output = args.output or ARTIFACTS / f"{args.experiment}.json"
    if args.command == "grade":
        report = RunReport.model_validate_json(args.input.read_text("utf-8"))
        await grade_report(
            report,
            SemanticJudge(
                args.judge_model,
                timeout=args.judge_timeout,
                reasoning_effort=args.judge_reasoning_effort,
            ),
            output,
            args.dataset,
            args.fixtures,
            args.corpus_root,
        )
        return int(any(c.status == "error" for c in report.cases))
    benchmark = load_benchmark(args.dataset, args.fixtures, args.corpus_root)
    settings = get_settings()
    embedding = CachedEmbeddings()
    connection = await connect_evaluation_database()
    try:
        if args.command == "prepare":
            repository_id = await prepare_index(connection, benchmark, embedding)
            print(f"Frozen evaluation index ready: {repository_id}")
            return 0
        repository_id, documents = corpus_documents(benchmark)
        if not await verify_index(connection, benchmark, repository_id, documents):
            raise ValueError(
                "Evaluation corpus is not prepared; run eval prepare first."
            )
        selected = set(args.case or [])
        questions = [
            q for q in benchmark.dataset.questions if not selected or q.id in selected
        ]
        if selected - {q.id for q in benchmark.dataset.questions}:
            raise ValueError("Unknown selected case ID.")
        retrieval_mode = (
            args.retrieval_mode if args.command == "retrieval" else "hybrid"
        )
        budget = args.candidate_budget if args.command == "retrieval" else None
        vector_k = (
            0 if retrieval_mode == "keyword" else budget or retrieval.VECTOR_CANDIDATES
        )
        keyword_k = (
            0 if retrieval_mode == "vector" else budget or retrieval.KEYWORD_CANDIDATES
        )
        rerank = args.rerank if args.rerank is not None else settings.repository_rerank
        reranker_config: dict[str, JsonValue] | None = None
        if rerank:
            # Download/load before timing cases; requests still have per-case limits.
            model = await asyncio.to_thread(get_reranker)
            reranker_config = {
                "model": RERANKER_MODEL,
                "revision": RERANKER_REVISION,
                "instruction": RERANKER_INSTRUCTION,
                "max_length": RERANKER_MAX_LENGTH,
                "batch_size": RERANKER_BATCH_SIZE,
                "device": str(model.device),
                "dtype": str(model.dtype),
                "score": "yes_minus_no_logit",
                "input": "file_path_newline_visible_parent_content",
                "placement": "all_expanded_deduplicated_parents_before_final_six",
                "truncation": "reject_oversize_no_truncation",
                "use_cache": False,
            }
        app_sources = sorted((ROOT / "apps/api/app").rglob("*.py"))
        metadata: dict[str, JsonValue] = {
            "experiment_name": args.experiment,
            "mode": args.command,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "architecture": "single_agent",
            "retrieval_mode": retrieval_mode,
            "dataset_version": benchmark.dataset.version,
            "dataset_sha256": digest(args.dataset.read_text("utf-8")),
            "dataset_case_count": len(benchmark.dataset.questions),
            "fixture_version": benchmark.fixtures.version,
            "fixture_sha256": digest(args.fixtures.read_text("utf-8")),
            "fixture_provenance": benchmark.fixtures.provenance,
            "corpus_commit_sha": benchmark.dataset.corpus_commit_sha,
            "corpus_repository": benchmark.dataset.repository,
            "corpus_index_id": repository_id,
            "corpus_file_count": len(benchmark.corpus),
            "corpus_chunk_count": len(documents),
            "application_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "application_sha256": digest(
                "\n".join(
                    str(p.relative_to(ROOT)) + ":" + digest(p.read_text("utf-8"))
                    for p in app_sources
                )
            ),
            "prompt_sha256": digest(
                CHAT_SYSTEM_PROMPT + inspect.getsource(agent_messages)
            ),
            "scorer_version": SCORER_VERSION,
            "scorer_sha256": digest(
                (Path(__file__).parent / "scoring.py").read_text("utf-8"),
            ),
            "model_configurations": {
                "chat": (
                    settings.hive_chat_model
                    if settings.hive_api_key
                    else settings.openai_chat_model
                ),
                "chat_temperature": "provider_default",
                "use_responses_api": not bool(settings.hive_api_key),
                "streaming": True,
                "embeddings": embedding.model,
                "dimensions": embedding.dimensions,
                "embedding_provider": settings.embeddings_provider,
                "embedding_profile": embedding.profile,
                "reranker": reranker_config,
            },
            "retrieval_parameters": {
                "vector_k": vector_k,
                "keyword_k": keyword_k,
                "final_k": retrieval.FINAL_PARENT_CONTEXTS,
                "rrf_k": retrieval.RRF_K,
                "parent_expansion": retrieval.PARENT_LINE_EXPANSION,
                "parent_max_lines": retrieval.PARENT_MAX_LINES,
                "parent_max_chars": retrieval.PARENT_MAX_CHARS,
                "chunk_tokens": CHUNK_TOKENS,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            "dependency_versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "langchain",
                    "langchain-openai",
                    "asyncpg",
                    "pydantic",
                )
            },
            "judge_configuration": None,
            "judge_calibrated": False,
            "case_timeout_seconds": args.timeout,
        }
        if settings.hive_api_key:
            model_configurations = metadata["model_configurations"]
            if isinstance(model_configurations, dict):
                model_configurations.update(
                    {
                        "chat_provider": "hive",
                        "chat_base_url": "https://api.thehive.ai/api/v3",
                        "chat_reasoning_effort": "high",
                        "chat_max_completion_tokens": 16384,
                        "chat_max_thinking_tokens": 8192,
                        "chat_timeout_seconds": 120,
                        "chat_max_retries": 1,
                    }
                )
        if rerank:
            versions = metadata["dependency_versions"]
            if isinstance(versions, dict):
                for name in ("sentence-transformers", "transformers", "torch"):
                    versions[name] = importlib.metadata.version(name)
        report = RunReport(metadata=metadata)
        write_report(report, output)
        for question in questions:
            case = CaseReport(
                case_id=question.id,
                category=question.category,
                question=question.question,
                metrics={name: Metric() for name in METRIC_NAMES},
            )
            records: list[ToolExecution] = []
            try:
                tools = (
                    await fixture_tools(
                        benchmark.fixtures,
                        question.fixture_scenario,
                        records,
                    )
                    if args.command == "full"
                    else []
                )
                runner: Runner = SingleAgentRunner(
                    connection,
                    embedding,
                    repository_id,
                    benchmark.dataset.corpus_commit_sha,
                    benchmark.dataset.repository,
                    tools,
                    records,
                    retrieval_only=args.command == "retrieval",
                    vector_k=vector_k,
                    keyword_k=keyword_k,
                    rerank=rerank,
                )
                case.result = await asyncio.wait_for(
                    runner.run(question.question),
                    timeout=args.timeout,
                )
                retrieval_metrics, case.partial_overlap = score_retrieval(
                    question,
                    case.result,
                    benchmark.corpus,
                    benchmark.dataset.corpus_commit_sha,
                )
                case.metrics.update(retrieval_metrics)
                if args.command == "full":
                    case.citations = referenced_citations(
                        case.result,
                        benchmark.corpus,
                        benchmark.dataset.corpus_commit_sha,
                        benchmark.dataset.repository,
                    )
                    case.metrics.update(
                        score_semantics(
                            question,
                            case.result,
                            case.citations,
                            None,
                            None,
                        )
                    )
                if any(record.fixture_miss for record in records):
                    raise ValueError(
                        "Unmatched fixture request; inspect execution records."
                    )
            except Exception as error:
                case.status, case.error = "error", f"execution: {type(error).__name__}"
                case.result.tool_executions = list(records)
                for name, metric in case.metrics.items():
                    if metric.status == "unscored":
                        case.metrics[name] = Metric(status="error")
            report.cases.append(case)
            write_report(report, output)
            print(f"Evaluated {question.id}: {case.status}", flush=True)
    finally:
        await connection.close()
    if args.command == "full" and args.judge_model:
        await grade_report(
            report,
            SemanticJudge(
                args.judge_model,
                timeout=args.judge_timeout,
                reasoning_effort=args.judge_reasoning_effort,
            ),
            output,
            args.dataset,
            args.fixtures,
            args.corpus_root,
        )
    print(f"Reports: {output} and {output.with_suffix('.md')}")
    return int(any(case.status == "error" for case in report.cases))


def main() -> int:
    """Parse explicit CLI commands; surface failures without printing credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "migrate", help="Apply existing migrations to the verified eval DB"
    )
    calibration = commands.add_parser(
        "calibrate", help="Check a judge before generation"
    )
    calibration.add_argument("--judge-model", required=True)
    calibration.add_argument("--timeout", type=float, default=60)
    calibration.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh")
    )
    calibration.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS / "judge-calibration-v1.json",
    )
    for name in ("prepare", "retrieval", "full", "grade"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", type=Path, default=DATA / "dataset.json")
        command.add_argument("--fixtures", type=Path, default=DATA / "github.json")
        command.add_argument("--corpus-root", type=Path, default=ROOT)
        command.add_argument("--experiment", default="single-agent-v1")
        command.add_argument("--output", type=Path)
        command.add_argument("--timeout", type=float, default=120)
        if name in {"retrieval", "full"}:
            command.add_argument("--case", action="append")
            command.add_argument(
                "--rerank",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Use local Qwen reranking; otherwise follow REPOSITORY_RERANKER",
            )
        if name == "retrieval":
            command.add_argument(
                "--retrieval-mode",
                choices=("hybrid", "vector", "keyword"),
                default="hybrid",
                help="Choose search branches; retain production parent expansion",
            )
            command.add_argument(
                "--candidate-budget",
                type=int,
                choices=(12, 24),
                help="Candidates per active branch; default uses production budgets",
            )
        if name in {"full", "grade"}:
            command.add_argument("--judge-model", required=name == "grade")
            command.add_argument("--judge-timeout", type=float, default=60)
            command.add_argument(
                "--judge-reasoning-effort",
                choices=("none", "low", "medium", "high", "xhigh"),
            )
        if name == "grade":
            command.add_argument("input", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("before", type=Path)
    comparison.add_argument("after", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    agreement = commands.add_parser("judge-agreement")
    agreement.add_argument("before", type=Path)
    agreement.add_argument("after", type=Path)
    agreement.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except ValidationError:
        print("Evaluation rejected: structured input failed validation.")
        return 1
    except ValueError as error:
        # Our validation errors contain only corpus/configuration context.
        print(f"Evaluation rejected: {error}")
        return 1
    except Exception as error:
        print(
            f"Evaluation failed ({type(error).__name__}); "
            "no complete baseline recorded."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
