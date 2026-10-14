# Historical maximum scores across selected experiments

Saved at the user's request as the report's historical best-results summary.
These are per-metric results across separate experiments, with provisional
partial-grading results retained. They do not represent one complete run or
the current Qwen-embedding/Hive-inference configuration with reranking disabled.

| Metric | Best recorded score | Scored cases | Evaluation coverage |
|---|---:|---:|---|
| Recall@6 | 82.35% | 17 | Completed retrieval |
| Precision@6 | 46.08% | 17 | Completed retrieval |
| Groundedness | 85.76% | 17 | Partial grading |
| Answer correctness | 58.82% | 17 | Partial grading |
| Factual completeness | 73.53% | 17 | Partial grading |
| Citation precision | 97.15% | 12 | Partial grading |
| Citation coverage | 36.52% | 17 | Partial grading |
| Citation validity | 100.00% | 16 | Deterministic citation checks |
| Correct abstention | 100.00% | 2 | Partial grading |
| Unnecessary abstention | 0.00% | 16 | Partial grading |

All values are macro averages over scored cases. The benchmark has 24
questions; 17 require code retrieval. Unnecessary abstention is lower-is-better.

Retrieval scores use OpenAI `text-embedding-3-small` embeddings and Qwen
reranking of expanded parent windows. All 24 retrieval cases completed.
The source is `retrieval-v8-qwen-parents.json`.

The partial semantic run generated answers with `gpt-4o-mini`, OpenAI
embeddings and Qwen reranking of child chunks. GPT-5.4 mini at medium reasoning
accepted 18/24 judgments; six failed cases are excluded from semantic scores.
Groundedness/correctness/completeness/coverage each score 17 cases because one
accepted response is N/A. Citation precision scores 12 cases. Correct abstention
scores only two of the four insufficient-evidence cases. Citation validity
is deterministic and includes 16 cases with referenced citations, independently
of judge failures. The valid source is `single-agent-v7-qwen-full-retry-utf8.json`;
the earlier encoding-corrupted retry is excluded.

## Suggested wording for the report

> Across selected experiments on a 24-question benchmark, RepoMind achieved
> reranked retrieval macro scores of 82.35% Recall@6 and 46.08% Precision@6
> over 17 applicable cases. A separate partially graded experiment yielded
> provisional groundedness of 85.76%, answer correctness of 58.82%, and factual
> completeness of 73.53% over 17 scored cases, with citation precision of 97.15%
> over 12 cases and citation coverage of 36.52% over 17 cases. Semantic grading
> accepted 18 of 24 cases. These figures are historical per-metric results
> across different configurations and are not a complete end-to-end baseline.

[Machine-readable scores, case contributions and provenance](data/maximum-scores-v1.json)

## Source reports

- [Completed reranked retrieval](../../../../.eval-artifacts/retrieval-v8-qwen-parents.md)
- [Partial semantic results](../../../../.eval-artifacts/single-agent-v7-qwen-full-retry-utf8.md)

Dataset version: `1`. Corpus commit: `678cd4ef010ef31f9293aa8ff0c92dafd255d002`.
Dataset, fixture and scorer identities match across the two selected sources.
No evaluations were rerun and no original artifacts or current settings were changed.
