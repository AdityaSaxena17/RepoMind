# Single-agent baseline, dataset v1

Measured on 2026-10-14. **24/24 cases completed and graded; 12/12 judge calibration examples passed.** There are no failed or unscored cases. GitHub findings refer to synthetic fixtures, not live GitHub history.

## Measured results

Scores are macro averages over applicable cases. N/A cases stay outside averages; there is no overall score. Answer correctness requires every reference fact and no material errors. Fact completeness reports partial success separately.

| Metric | Macro score | Scored cases | N/A |
|---|---:|---:|---:|
| Recall@6 | 51.47% | 17 | 7 |
| Precision@6 | 21.57% | 17 | 7 |
| Groundedness | 74.78% | 23 | 1 |
| Answer correctness | 22.73% | 22 | 2 |
| Fact completeness | 36.74% | 22 | 2 |
| Citation precision | 71.43% | 14 | 10 |
| Citation coverage | 4.80% | 23 | 1 |
| Citation validity | 82.14% | 14 | 10 |
| Correct abstention | 75.00% | 4 | 20 |
| Unnecessary abstention (lower is better) | 5.00% | 20 | 4 |

| Category | Cases | Fully correct answers | Fact completeness |
|---|---:|---:|---:|
| architecture | 6 | 0/6 applicable | 5.56% |
| github | 4 | 4/4 applicable | 100.00% |
| insufficient | 4 | 1/2 applicable | 50.00% |
| mixed | 4 | 0/4 applicable | 31.25% |
| single_file | 6 | 0/6 applicable | 25.00% |

Retrieval applies to 17 cases; seven have no required code evidence. Primary recall recovered 21/40 evidence units, and precision credited 22/102 returned windows. These pooled counts differ from the macro scores above. A lack of references makes citation precision/validity N/A; it does not make citation coverage perfect.

## Concrete failures retained

- `code-01`: retrieval missed the required worker evidence. The answer stated 1,000,000 bytes instead of 1,048,576 and invented exclusion examples from unrelated ignore-file context.
- `code-02`: the answer omitted the verified tokenizer/chunk-size/overlap configuration and described character indexing as byte offsets.
- `code-04`: the answer missed the 250-line cap and incorrectly characterized the 16,000-character truncation rule.
- `mixed-01`: the answer found the unused-card issue but explained client-side handling instead of the server loop that emits cards for retrieved documents.
- `mixed-02`: the answer described concurrency as a proposed new feature although the frozen corpus already uses asyncio.gather. The runtime connection fix intentionally differs from that frozen source snapshot.
- `mixed-03`: the answer did not establish that the fixture file exceeds the indexed worker limit and introduced alternative decoding causes.
- `mixed-04`: the answer confused assistant-message failure handling and a UI readiness check with the worker failure state and server readiness gate.

Citation coverage is especially weak: most domain claims lack supporting inline references. Displayed source cards do not earn citation credit. The synthetic GitHub cases were fully correct (4/4); all mixed cases missed required cross-source facts. These results were preserved without changing production prompts or tuning retrieval.

## Frozen configuration

- Corpus: `AdityaSaxena17/RepoMind` at `678cd4ef010ef31f9293aa8ff0c92dafd255d002`; 101 files and 153 chunks.
- Dataset/fixtures/scorer versions: `1` / `1` / `1`.
- Generator: `gpt-4o-mini`, provider-default temperature, Responses API, production streaming assembly, empty conversation history.
- Embeddings: `text-embedding-3-small`, 1536 dimensions, real provider vectors with a content-addressed cache.
- Judge: `gpt-5.4-2026-03-05`, low reasoning, no sampling temperature, 120-second request timeout, one provider retry and at most two grounding-validation corrections. At most three cases are graded concurrently.
- Application Git revision: `678cd4ef010ef31f9293aa8ff0c92dafd255d002`. The evaluated working tree included the connection fix and shared generation extraction; its source hash below distinguishes it from the commit alone.
- Database: dedicated `repomind_eval` database/login, existing migrations and pgvector. A read-only privilege query confirmed the login has no write privileges on application tables.

```json
{
  "dataset_sha256": "0f1e1a8d5a7d741901a8e8d28a0673b34b6e53c5a9b190762283a67070a1a3ef",
  "fixture_sha256": "8bb9324fb67214514a9964c00135fc577f30634f195bd221ae83024f81281a9d",
  "application_sha256": "6619dffc5f5587bf476c2b6f6ed49caf0ab0a96b5221d446e50f26bba4326b56",
  "prompt_sha256": "46d44bdbfe8bf81250c272410faa9f6cdf89aa6600fb144fcd4283c305e5eefa",
  "scorer_sha256": "2002a69e91e4988978577009691ac60f888e1ad0347c737b2fac7eb326242cfb",
  "retrieval_parameters": {
    "vector_k": 12,
    "keyword_k": 12,
    "final_k": 6,
    "rrf_k": 60,
    "parent_expansion": 80,
    "parent_max_lines": 250,
    "parent_max_chars": 16000,
    "chunk_tokens": 800,
    "chunk_overlap": 100
  },
  "judge_configuration": {
    "model": "gpt-5.4-2026-03-05",
    "temperature": null,
    "reasoning_effort": "low",
    "timeout_seconds": 120.0,
    "max_retries": 1,
    "grounding_validation_retries": 2,
    "structured_output": "json_schema",
    "rubric_sha256": "b6f1007ff2768398b1182601e453e9ac060aed8faa9e6062d83066b371753f69",
    "implementation_sha256": "8ecae43d49ad2dac0e0c3ad497178a237cb30736390f10da54cbd922b40e083e",
    "schema_sha256": "e4b079d2d5b29ccdca72603d283f39f92af14b790b3fafc2f2b7a992283f85ce",
    "calibration_sha256": "148fd925b2b25649a5ef7f078024305bdc16f23f0a5682f4c77bfee9483e705e"
  }
}
```

## Artifacts and verification

- [Versioned compact results, answers and correctness judgments](data/baseline-v1.json).
- [Full JSON with evidence, citations, tool records and judgments](../../../../.eval-artifacts/single-agent-v1-accepted.json).
- [Full Markdown report](../../../../.eval-artifacts/single-agent-v1-accepted.md).
- [Retrieval JSON](../../../../.eval-artifacts/single-agent-v1-retrieval-accepted.json) and [Markdown](../../../../.eval-artifacts/single-agent-v1-retrieval-accepted.md).

The full reports and embedding cache are local, ignored artifacts. Their file hashes are retained in the compact results. The answers were captured once in `single-agent-v1-final` and then graded from the saved artifact. Earlier incomplete harness runs and failed judge calibrations remain in `.eval-artifacts/`; they are excluded from the accepted baseline.

Actually executed checks:

- Offline pytest: **58 passed, 1 skipped** (database test is opt-in).
- Opt-in PostgreSQL integration: **1 passed**, using separate deterministic embeddings for mechanics only.
- Ruff, `mypy app` (18 modules), and `pnpm --filter api build`: passed.
- Evaluation migration command: verified target, five existing migrations, none pending.
- Comparison command self-check: passed, with zero paired retrieval deltas.

Retrieval and full quality evaluation were actually executed; these scores are not unit-test results. The annotations were source-reviewed during implementation, without independent human sign-off. Calibration checks 12 known examples and does not guarantee human agreement on every claim. The benchmark is small, uses one frozen repository and synthetic GitHub scenarios, and does not establish statistical significance.

Repeat the setup and capture/grade commands in [README.md](README.md). To compare future architectures fairly, retain these dataset, corpus, fixture, scorer and judge identities; expose intentional application/model changes. Regrade saved full-run artifacts, not this compact summary.
