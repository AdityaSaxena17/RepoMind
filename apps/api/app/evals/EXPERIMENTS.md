# Retrieval improvements and cheaper judge trials

Measured on 2026-10-14. The original [baseline](BASELINE.md) and its GPT-5.4
grades are preserved. Dataset, corpus commit, fixtures, scorer, reference facts,
generation prompt and embedding model are unchanged. These are descriptive
results on the initial benchmark, not a statistical significance claim.
The latest selected configuration has Recall@6 **80.88%** and Precision@6
**38.24%**, using tighter identifier boosts and 24 candidates per branch. Earlier
sections retain the preceding experiments; the identifier/budget section records
the latest changes.

The subsequent opt-in parent-window Qwen experiment reached **82.35%** recall
and **46.08%** precision. The no-reranker configuration remains the application
default. See the parent-window section for its separate retrieval measurements.

## Executed retrieval experiments

Every experiment completed all 24 questions. Retrieval applies to 17 cases; seven
have no required code evidence. Scores are macro averages, scored independently
of generation and LLM judging.
The "lexical" labels below describe changes to the keyword branch within hybrid
retrieval. They do not mean vector search was disabled.

| Experiment | Recall@6 | Precision@6 |
|---|---:|---:|
| Original baseline | 51.47% | 21.57% |
| Lexical OR, identifier/path priority | 65.69% | 23.53% |
| Parent selection only | 51.47% | 21.57% |
| Lexical + parent selection | 65.69% | 23.53% |
| Lexical + distinct term coverage | 71.57% | 24.51% |
| Corrected chunk line ranges only | 55.39% | 26.47% |
| Corrected ranges + lexical | 70.59% | 32.35% |
| **Corrected ranges + lexical + term coverage: selected at this stage** | **76.47%** | **33.33%** |
| Corrected ranges + lexical + term coverage + parent selection | 74.51% | 29.41% |

The selected implementation recovers **28/40 required evidence units**, versus
21/40, and credits **34/102 returned windows**, versus 22/102. Complete-span recall
and all original scoring conventions are preserved. Macro recall improves by
25.00 percentage points and precision by 11.76 points. The versioned
[JSON summary](data/retrieval-experiments-v1.json) records configurations, source
and artifact hashes, categories, all per-case scores, and the range diagnosis.
All 216 saved case results were replayed against the unchanged deterministic scorer.

The retained change has three parts:

- Keyword search uses OR over meaningful original token spellings. PostgreSQL's
  English dictionary only identifies stopwords; the existing `simple` index stays
  unstemmed. Explicit identifiers and paths rank first, then distinct query-term
  coverage, then the existing text rank. Parameterized inputs, empty queries and
  stopword-only queries are covered. C collation makes keyword ties reproducible.
- Distinct term coverage prevents frequent generic words from dominating. In the
  ingestion question, a chunk matching only repeated `GitHub` ranked above chunks
  covering several relevant terms. No paid reranker or query-rewriting model is used.
- The installed token splitter sometimes reports `start_index=-1`: its offset
  search subtracts token overlap as characters. The worker previously treated this
  as a real offset, yielding line 1 for later chunks. The fix locates the unchanged
  text when that search fails and counts actual LF/CRLF separators. Twenty of 153
  source ranges changed; every chunk text, content hash and embedding input stayed
  identical. LF/CRLF regressions failed before the fix and passed afterward.

The evaluation index with corrected ranges has a new content-addressed identity,
`eval_2841872174440f5f65b780f9409d6a869cff480832627fce611d6097a8c06ce0`.
Preparation reused cached embeddings and preserved the old index. This is a
metadata repair, with no embedding-model migration. **Existing application indexes
need rebuilding to receive the corrected ranges.** No application data was modified.

At this stage, vector and keyword budgets were 12 each; RRF remains 60 and six final contexts
retain the existing line/character limits. Parent selection remains unchanged:
its additional complexity did not improve both scores, even after the range fix.
The tested variant and its offline regressions are retained as an
[experimental patch](data/parent-selection-experiment.patch), not enabled at runtime.
That archived patch targets the pre-ablation lexical snapshot, not the current source.

Failures from this stage remain visible. `architecture-02` regresses from 0.50 to 0.25 recall,
although final precision remains 0.33. `code-06`, `architecture-04` and
`architecture-05` still recover only part of the required evidence. Future tuning
should use candidate/final-window diagnostics and additional reviewed questions;
the current dataset is too small to establish general gains by itself.

## Luna trial and review

Luna graded the same 24 saved baseline answers with `gpt-5.6-luna`, medium
reasoning and a 120-second request timeout. All 12 calibration examples passed.
No answers were regenerated for this judge trial.

| Adoption check against the preserved GPT-5.4 baseline | Result |
|---|---|
| Correctness case outcomes | Match |
| Abstention outcomes | Differ on `mixed-03` |
| Reference-fact label agreement, required >=95% | 39/43 = 90.70% |
| Groundedness macro difference, allowed <=5 points | +3.32 points |
| Citation precision macro difference, allowed <=5 points | +6.61 points |
| Citation coverage macro difference, allowed <=5 points | +12.91 points |

Luna fails the predeclared adoption checks. Its results remain an experiment.
Agreement is not accuracy, and review also found weaknesses in the original judge:

- `code-04/f2`: the old judge labels the fact matched while its own reason says
  the 16,000-character limit is missing. Luna's missing label is more consistent.
- `architecture-01/f1`: the answer reverses job creation/enqueue ordering. The
  old judge catches the contradiction; Luna labels the fact missing.
- `mixed-03/f3`: the answer hedges about decoding causes rather than establishing
  the size cutoff. Judges differ on contradiction versus omission and whether the
  hedging constitutes unnecessary abstention.
- `mixed-04/f1`: the answer gives the availability motivation but omits the
  transactional-replacement proposal. Luna's missing label is stricter than the
  original matched label.
- Citation coverage differences largely concern attachment scope. Luna attaches
  end-of-answer references to many preceding claims in `architecture-05` and
  `github-03`. In `github-04`, the returned file URL is valid and its tool payload
  supports the change counts; the judges disagree on which list claims that link
  cites. These are rubric interpretation differences, not evidence of fabricated
  URLs in that case. Prompts, calibration labels and scores were not altered.

The `judge-agreement` command checks identical saved answers, evidence and rubric
hashes, matches stable reference-fact IDs, and reports per-case differences and
the adoption gates. The architecture `compare` command still requires matching
judge configurations.

## Requested GPT-5.4 mini comparison

The user's subsequent instruction selects GPT-5.4 mini for both answer sets and
prioritizes retrieval. Both runs used `gpt-5.4-mini-2026-03-17`, medium reasoning,
the same unchanged judge prompts/schemas, and a 120-second request timeout.
All 12 calibration examples passed before each grading run.

The original 24 baseline answers were regraded without regeneration. The selected
retrieval candidate generated all 24 answers successfully with the same production
generator and fixed fixtures, then received mini grading. No fixture misses or
generation failures occurred in this selected run.

| Mini grading | Valid judgments | Rejected judgments | Complete quality baseline |
|---|---:|---:|---|
| Saved baseline, after one retry of failures | 20/24 | 4 | No |
| Selected retrieval candidate | 20/24 | 4 | No |

The mini judge repeatedly supplied claim or limitation quotations that do not
occur verbatim in the answer. Validation rejected these labels after the existing
bounded corrections. A single additional retry of only the five initial baseline
failures recovered `mixed-04`; the raw first attempt and retry are preserved, and
answers, evidence and successful grades were unchanged. Further paid retries were
stopped to keep this iteration focused on retrieval.

Remaining baseline grading errors: `code-04`, `missing-02`, `missing-04`,
`mixed-03`. Candidate grading errors: `code-02`, `architecture-05`, `missing-01`,
`missing-02`. These are judge protocol errors, not zero answer-quality scores or
measured abstentions. Passing the small calibration set did not establish reliable
quotation handling on the longer benchmark answers.

The strict comparison command accepted the matching mini configurations and
produced paired descriptive results. Grounding/correctness have 17 scored pairs;
correct abstention has only one. **This is an incomplete semantic comparison and
does not replace the original accepted quality baseline.** The fully measured
retrieval gains above do not depend on mini judgments.

The versioned [judge summary](data/judge-experiments-v1.json) records model settings,
calibration counts, artifact hashes, rejected cases and paired metrics. Full local
reports retain every answer, source context, tool execution and accepted judgment:

- [Mini baseline Markdown](../../../../.eval-artifacts/single-agent-v1-mini-final.md)
  and [JSON](../../../../.eval-artifacts/single-agent-v1-mini-final.json).
- [Selected candidate Markdown](../../../../.eval-artifacts/single-agent-v4-selected.md)
  and [JSON](../../../../.eval-artifacts/single-agent-v4-selected.json).
- [Paired comparison JSON](../../../../.eval-artifacts/mini-retrieval-comparison.json).

The earlier full capture with only lexical changes and its execution retry remain in local
artifacts. Its GPT-5.4 low-reasoning grading attempts failed partial-citation
calibration twice; neither was accepted. The later selected run supersedes that
capture for retrieval evaluation, and no full GPT-5.4 judge calls were made after
the user's mini instruction.

## Vector-only, keyword-only and hybrid comparison

All three modes were executed on 2026-10-14 against the same corrected frozen
index, embeddings, dataset, fixtures and scorer. All 24 cases completed without
errors in each mode; 17 cases have applicable retrieval metrics and seven are N/A.
No answers were generated and no judge calls were made.

| Mode | Candidate budget | Recall@6 | Precision@6 | Recovered units | Relevant windows |
|---|---|---:|---:|---:|---:|
| Vector only | 12 vector | 52.45% | 25.49% | 22/40 | 26/102 |
| Keyword only | 12 keyword | 67.16% | 32.35% | 24/40 | 33/102 |
| **Hybrid** | 12 vector + 12 keyword | **76.47%** | **33.33%** | **28/40** | **34/102** |

Each mode uses the production parent expansion and six final contexts. The
candidate budgets differ because the experiment disables one existing branch;
this is not an equal-total-candidate-budget comparison. Macro recall differs from
pooled recovered-unit fractions. All 72 saved results were replayed against the
unchanged scorer. Hybrid's final contexts exactly match the previous selected run.

Hybrid remains the production default. Vector search adds useful evidence even
though it is weaker alone: `code-02` recall is 0.50 with keywords and 1.00 with
vectors or hybrid; `mixed-04` is 0.00 with keywords and 0.33 with vectors or hybrid.
For `code-06`, neither single branch completes a required evidence unit alone,
while their combined final windows recover half of the units.

Fusion still loses evidence in `architecture-02`: vector-only recall is 0.50,
keyword-only is 0.00, and hybrid is 0.25. This is a concrete remaining selection
failure; the aggregate gains do not establish that hybrid wins on every question.
This experiment does not compare parent documents against child-only retrieval.

The versioned [mode comparison](data/retrieval-modes-v1.json) contains configuration,
artifact hashes, category results and per-case scores. Full local reports:

- [Vector Markdown](../../../../.eval-artifacts/retrieval-v5-vector-only.md) and
  [JSON](../../../../.eval-artifacts/retrieval-v5-vector-only.json).
- [Keyword Markdown](../../../../.eval-artifacts/retrieval-v5-keyword-only.md) and
  [JSON](../../../../.eval-artifacts/retrieval-v5-keyword-only.json).
- [Hybrid Markdown](../../../../.eval-artifacts/retrieval-v5-hybrid.md) and
  [JSON](../../../../.eval-artifacts/retrieval-v5-hybrid.json).

## Tighter identifier boosts, then increased candidate budgets

These experiments were executed in the requested order on 2026-10-14. Qwen was
excluded from the retrieval implementation and benchmark. The existing corpus,
index, dataset, fixtures, scorer, embeddings, RRF constant and parent-expansion
rules stayed fixed. The final count remains six contexts.

| Stage | Candidates | Recall@6 | Precision@6 | Recovered units | Relevant windows |
|---|---|---:|---:|---:|---:|
| Previous hybrid baseline | 12 vector + 12 keyword | 76.47% | 33.33% | 28/40 | 34/102 |
| Tighter identifier boosting | 12 vector + 12 keyword | 80.88% | 36.27% | 31/40 | 37/102 |
| **Tighter boosting + larger pool: selected** | **24 vector + 24 keyword** | **80.88%** | **38.24%** | **31/40** | **39/102** |

Tighter boosting improved both scores before the larger pool was tested. The
keyword branch now reserves identifier boosts for backticked names, snake_case,
and explicit calls such as `prefixHandler()`. Bare CamelCase words, including
`GitHub` and `RepoMind`, remain ordinary searchable terms. File-path boosts,
literal identifier boundaries and parameterized SQL are preserved.

At 12+12, `architecture-02` improves from 0.25 to 1.00 recall, recovering all four
required evidence units. Its precision improves from 0.33 to 0.50. `code-05` and
`architecture-06` also gain relevant windows without changing recall.

Increasing to 24+24 adds two relevant windows overall while preserving every
case's recall. Precision improves in `code-01`, `architecture-02` and
`architecture-03`. The retained regression is `code-05`: precision falls from
0.50 to 0.33 relative to the tighter 12+12 experiment, matching its older baseline;
its recall stays 1.00. These results support the selected defaults on this small
benchmark and do not establish gains on unseen questions or repositories.

Production defaults now use 24 candidates per branch. Retrieval-only runs can
override them with `--candidate-budget 12` or `24`; actual budgets are recorded in
the report. A separate default-settings run completed all 24 cases and returned
exactly the same final contexts as the explicit 24+24 experiment. All 96 saved
case results across the baseline, two experiments and default confirmation were
replayed against the unchanged scorer. Retrieval is applicable to 17 cases, with
seven N/A cases and no failed cases in the complete runs. No answer generation or
semantic grading was executed for this iteration.

The initial tighter-boost run stopped after saving 22 cases when report
checkpointing raised `PermissionError`. Its partial JSON/Markdown artifacts are
preserved and excluded from the comparison; the complete rerun is separate.

The versioned [identifier/budget summary](data/identifier-budget-v1.json) records
configuration and artifact hashes, category scores, per-case results and the
interrupted attempt. Full local reports:

- [Tighter 12+12 Markdown](../../../../.eval-artifacts/retrieval-v6-tight-identifiers-final.md)
  and [JSON](../../../../.eval-artifacts/retrieval-v6-tight-identifiers-final.json).
- [Tighter 24+24 Markdown](../../../../.eval-artifacts/retrieval-v6-tight-identifiers-24.md)
  and [JSON](../../../../.eval-artifacts/retrieval-v6-tight-identifiers-24.json).
- [Selected-default Markdown](../../../../.eval-artifacts/retrieval-v6-selected.md)
  and [JSON](../../../../.eval-artifacts/retrieval-v6-selected.json).

## Dedicated Qwen reranker experiment

The official `Qwen/Qwen3-Reranker-0.6B` was executed through Sentence Transformers
6.1.0, pinned to model revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`.
PyTorch 2.10.0+cu128 used the RTX 3050 in float16. A separate real forward-pass
check observed both transformer weights and input IDs on `cuda:0`; NVIDIA also
reported active GPU computation. Tokenization and database operations still use
the CPU, and remote generation/judging leave gaps in GPU activity.

The model scored all deduplicated fused children, using path plus chunk content
and trained yes-minus-no logits, before the existing parent selection. The code
instruction, batch size four and 8,192-token limit are recorded in the report.
No embeddings, gold annotations, corpus, fixture, scorer, candidate budgets or
parent-window rules changed.

| Configuration | Recall@6 | Precision@6 | Recovered units | Relevant windows |
|---|---:|---:|---:|---:|
| Existing hybrid, 24+24 | **80.88%** | **38.24%** | **31/40** | **39/102** |
| Qwen reranked, 24+24 | 75.00% | 34.31% | 30/40 | 35/102 |

All 24 retrieval cases completed without errors; 17 are applicable and seven N/A.
Macro recall decreased 5.88 percentage points and precision decreased 3.92 points.
The feature remains opt-in with `REPOSITORY_RERANKER=none` by default.

Retained regressions include `code-02`, where the selected windows omit the
chunk-size/overlap constants; `architecture-02`, where chunk persistence evidence
is lost; and `mixed-01`, where UI/card passages displace the backend citation-card
emission code. `architecture-03` and `mixed-03` improve, but do not offset these
losses. This is evidence against adopting this particular configuration on the
current benchmark, not a claim about all rerankers or unseen repositories.

The initial CPU evaluation was stopped after two completed cases and is preserved
separately. Its partial results are excluded. An initial judge calibration attempt
failed with API connection errors on all examples; after connection probes passed,
GPT-5.4 mini at medium reasoning passed all 12 examples in a separate artifact.

Full local retrieval reports:

- [Qwen Markdown](../../../../.eval-artifacts/retrieval-v7-qwen-cuda.md) and
  [JSON](../../../../.eval-artifacts/retrieval-v7-qwen-cuda.json).
- [Paired comparison](../../../../.eval-artifacts/qwen-v7-retrieval-comparison.json).

### Full generation and semantic grading

All 24 answers were generated successfully with the shared production agent,
`gpt-4o-mini`, deterministic GitHub fixtures and the GPU reranker. Their final
contexts exactly match the retrieval-only run. GPT-5.4 mini at medium reasoning
passed the full run's 12-example calibration, using the unchanged prompts,
schemas and scorer.

The first grading pass accepted 18 cases. Six cases failed: `architecture-01`
(claim quotation validation), `architecture-02` and `architecture-03` (timeouts),
`missing-02` and `missing-04` (limitation quotation validation), and `mixed-04`
(an evidence ID never consumed). No failed labels earned scores.

An intermediate retry helper used Windows' default encoding to read UTF-8,
corrupting some saved answer/context text. That retry artifact is invalid and
excluded entirely. The corrected helper explicitly read UTF-8, validated every
context against the frozen corpus, and verified exact original answer/evidence
equality before and after grading. All six corrected retries received HTTP 429
errors. The 18 successful original judgments were retained unchanged, and the
first pass and all failed attempts remain preserved.

| Metric | Macro score | Scored cases |
|---|---:|---:|
| Recall@6 | 75.00% | 17 |
| Precision@6 | 34.31% | 17 |
| Groundedness | 85.76% | 17 |
| Answer correctness | 58.82% | 17 |
| Factual completeness | 73.53% | 17 |
| Citation precision | 97.15% | 12 |
| Citation coverage | 36.52% | 17 |
| Citation validity | 100.00% | 16 |
| Correct abstention | 100.00% | 2 of 4 insufficient-evidence cases |
| Unnecessary abstention | 0.00% | 16 answerable cases |

Semantic aggregates are provisional and exclude six failed judgments. Citation
validity is deterministic: all 43 referenced citations are valid, while eight
cases have no references and are N/A. It does not establish citation support.
No complete semantic quality baseline or semantic before/after gain is claimed.
The unchanged original accepted baseline remains separate.

The versioned [Qwen summary](data/qwen-reranker-v1.json) records artifact hashes,
configurations, per-case metrics, failures and the discarded retry. All 96 saved
case results across the baseline, retrieval, first full pass and corrected retry
were replayed against the unchanged retrieval scorer. Accepted semantic labels
and numerical scores were also revalidated. Full local reports:

- [First full pass Markdown](../../../../.eval-artifacts/single-agent-v7-qwen-full.md)
  and [JSON](../../../../.eval-artifacts/single-agent-v7-qwen-full.json).
- [Corrected retry Markdown](../../../../.eval-artifacts/single-agent-v7-qwen-full-retry-utf8.md)
  and [JSON](../../../../.eval-artifacts/single-agent-v7-qwen-full-retry-utf8.json).

After the integration, **93 offline tests passed** and one opt-in database test
was skipped. The explicit PostgreSQL integration test separately passed with
the final dependencies. Ruff, `mypy app` (19 source files), and the API compile
build passed. These checks are separate from the executed quality evaluations.

## Parent-window Qwen reranking

The reranker now sees the expanded parent windows that can become final contexts.
Vector SQL, keyword SQL, RRF, candidate budgets (24 per branch), embeddings, corpus,
annotations and scorer are unchanged. The full fused pool is expanded and merged
under the existing 250-line/16,000-character limits before visible-window
deduplication. Qwen scores every remaining parent and selects six; citation source
indices are reassigned after selection. Distinct windows in one file remain eligible.
Disabled reranking retains the previous selection behavior.

The same official Qwen weights, pinned revision, instruction and batch size are
used. The token limit increases from 8,192 to the model's supported 32,768 tokens
to accommodate parents. Tokenizer truncation is disabled and oversized inputs
fail explicitly. KV caching is disabled because relevance scoring needs a single
forward pass. The actual run recorded CUDA float16 on the RTX 3050; independent
`nvidia-smi` sampling observed 100% GPU utilization and 2,275 MiB GPU memory use.

| Experiment | Recall@6 | Precision@6 | Recovered units | Relevant windows |
|---|---:|---:|---:|---:|
| No reranker, selected hybrid baseline | 80.88% | 38.24% | 31/40 | 39/102 |
| Qwen reranks child chunks | 75.00% | 34.31% | 30/40 | 35/102 |
| **Qwen reranks expanded parents** | **82.35%** | **46.08%** | **31/40** | **47/102** |

All three experiments completed 24 questions; 17 have applicable code evidence
and seven are N/A. Parent reranking improves macro recall by 7.35 percentage
points and precision by 11.76 points over child reranking. Against no reranking,
the gains are 1.47 and 7.84 points. Total recovered units remain 31/40; the macro
recall gain reflects their distribution across questions with different numbers
of required units, rather than a larger recovered-unit count.

Relative to no reranking, `code-06` and `mixed-03` recover additional evidence,
while `code-02` still loses the chunk-size/overlap constants and `architecture-02`
loses a persistence unit. Parent reranking restores `mixed-01`'s backend citation
emission evidence lost by child reranking. It improves precision in several
architecture and mixed cases, but does not solve every omission. These are
descriptive results on this benchmark, without a statistical significance claim.

The versioned [parent-window summary](data/qwen-parent-reranker-v1.json) preserves
artifact hashes, configurations, per-case metrics, overlap fractions and selected
source ranges. All 72 case results across the three reports were replayed against
the frozen corpus and unchanged deterministic scorer. Paired comparisons require
matching dataset, corpus, fixture, scorer and judge configurations. The original
accepted full-baseline artifact's SHA-256 remains unchanged.

Reports:

- [Parent retrieval Markdown](../../../../.eval-artifacts/retrieval-v8-qwen-parents.md)
  and [JSON](../../../../.eval-artifacts/retrieval-v8-qwen-parents.json).
- [Comparison against no reranking](../../../../.eval-artifacts/qwen-v8-vs-retrieval-v6-selected.json).
- [Comparison against child reranking](../../../../.eval-artifacts/qwen-v8-vs-retrieval-v7-qwen-cuda.json).

The run used cached embeddings and the local reranker. No answer generation or
semantic grading was executed for this variant; those metrics remain unscored.
Reranking stays opt-in pending a complete full-RAG evaluation.

After this change, **96 offline tests passed** and one opt-in integration test
was skipped. The explicit PostgreSQL integration test separately **passed**.
Regressions cover expanded text, candidates beyond the previous 18-child bound,
duplicate/empty windows, stable ties, distinct same-file windows, visible trimmed
text, oversized-token rejection and error propagation. Ruff, `mypy app` (19 source
files), and the API compile build passed. Tests and integration mechanics are
separate from the executed retrieval quality benchmark.

## Hive-hosted DeepSeek evaluation judge

The production agent remains on LangChain. The optional evaluation judge now
supports `deepseek-ai/deepseek-v4.1-flash` through Hive's documented
OpenAI-compatible endpoint, using `openai.AsyncOpenAI` directly and the configured
`DEEPSEEK_API_KEY` at the time (a Hive key; now renamed `HIVE_API_KEY`). The earlier direct-DeepSeek HTTP 401 responses
were caused by sending that provider's key to the wrong endpoint. A live Hive
probe successfully returned non-streaming JSON with high reasoning.

Hive documents the OpenAI SDK path and its reasoning parameters. LangChain's
`ChatOpenAI` targets the official OpenAI API and does not preserve every
third-party response extension. Using the SDK directly for this judge keeps its
request and response validation explicit without changing production inference.
The SDK version was already installed transitively; it is now declared directly,
with the lockfile updated offline and no SDK version upgrade.

References: [Hive model/API documentation](https://docs.thehive.ai/docs/deepseek-v41-flash),
[LangChain API scope](https://docs.langchain.com/oss/python/integrations/chat/openai),
[OpenAI Python SDK documentation](https://developers.openai.com/api/reference/python).

High reasoning, two isolated judge tasks, reference facts, the 12 calibration
examples and the numerical scorer are retained. Hive's JSON mode is followed by
full-string Pydantic validation, exact-quotation/evidence-ID checks, and the
existing semantic consistency checks. Empty, malformed and incomplete responses
fail explicitly. Each request has a 120-second timeout and one provider retry;
clients close on both success and failure.

The initial 16,384-token completion profile passed all 12 small calibration
examples but accepted only 21/24 full-answer judgments. `code-02`, `code-06` and
`architecture-02` returned empty/incomplete responses after validation retries.
This attempt is preserved separately. A diagnostic on `code-02` used 10,910
reasoning tokens out of 13,459 completion tokens. Hive's documented thinking
budget produced a complete, valid judgment on the same case with
`max_thinking_tokens=8192`. The final profile retains **high reasoning** and the
16,384-token completion cap while explicitly reserving space for JSON labels.
All 12 calibration examples passed again under that configuration.

Hive has its own billing. Its pricing page listed discounted $0.12/M input,
$0.48/M output and $0.0024/M cache reads on 2026-10-14. DeepSeek's direct-API
off-peak schedule is not assumed to apply to Hive. Check
[Hive's current pricing](https://thehive.ai/pricing) before future experiments.

Only the saved original baseline was regraded under this profile. **All 24 cases
received valid judgments; there were no failures**, and all 12 calibration
examples passed within the run. The
second regrading/comparison run was declined. The newer parent-window variant
still has retrieval-only results. Original GPT artifacts and generated answers
remain unchanged. The strict `judge-agreement` command requires matching grader
implementation hashes; extracting provider invocation changed that hash, so no
cross-implementation GPT/Hive agreement result is claimed or forced by rewriting
compatibility metadata.

### Original saved baseline, regraded with Hive

These scores describe the historical original answers and contexts, rather than
answers generated with the later retrieval improvements. Its retrieval remains
51.47% recall and 21.57% precision; the separate parent-window experiment remains
82.35% and 46.08%. Regrading does not measure a change in RepoMind's answers.

| Metric | Macro score | Scored cases |
|---|---:|---:|
| Recall@6 | 51.47% | 17 |
| Precision@6 | 21.57% | 17 |
| Groundedness | 80.04% | 22 |
| Answer correctness | 22.73% | 22 |
| Factual completeness | 34.85% | 22 |
| Citation precision | 78.24% | 14 |
| Citation coverage | 25.59% | 22 |
| Citation validity | 82.14% | 14 |
| Correct abstention | 75.00% | 4 |
| Unnecessary abstention | 0.00% | 20 |

Two cases have no reference facts or domain claims, so those generation metrics
are N/A. Ten cases have no referenced citations and are N/A for citation precision
and validity. All applicability and failure counts remain explicit in the report.

The [versioned Hive summary](data/hive-deepseek-judge-v1.json) records configuration,
per-case metrics and fact labels, hashes, the retained initial attempt and budget
diagnostic. All 24 final results were replayed against the unchanged retrieval
scorer, and their semantic labels/numerical scores were revalidated. Every saved
answer and consumed evidence item exactly matches the original artifact and the
frozen corpus; the original GPT baseline's file hash is unchanged.

Source review of five differing fact labels in `code-04`, `architecture-05`,
`mixed-03` and `mixed-04` found the Hive decisions consistent with the saved
answers and required fact components. The notes are recorded without editing
labels. This is implementation-assistant review, not independent human sign-off
or a compatible cross-judge agreement result.

Local reports:

- [Final Hive Markdown](../../../../.eval-artifacts/single-agent-v1-hive-deepseek-high-v2.md)
  and [JSON](../../../../.eval-artifacts/single-agent-v1-hive-deepseek-high-v2.json).
- [Final calibration](../../../../.eval-artifacts/hive-deepseek-high-calibration-v2.json).
- [Initial 21/24 attempt](../../../../.eval-artifacts/single-agent-v1-hive-deepseek-high.json).
- [Thinking-budget diagnostic](../../../../.eval-artifacts/diagnose_hive_budget.json).

Verification: **107 offline tests passed**, one default integration test was
skipped, and the explicit PostgreSQL integration test separately **passed**.
Ruff, `mypy app` (19 files), formatter checks and the API compile build passed.
Offline regressions exercise the actual SDK transport with mock HTTP responses:
credential/endpoint selection, high reasoning and token budgets, JSON payloads,
empty/malformed/truncated responses, provider errors/retries and client cleanup.
Both providers also share tests for gold/evidence separation and bounded
quotation-validation retries. Ordinary tests make no paid calls.

### Shared Hive credential and production generation (2026-10-14)

The local credential was renamed to `HIVE_API_KEY`, which is now used for both
LangChain production/evaluation answer generation and the SDK-based Hive judge.
The old `DEEPSEEK_API_KEY` name is no longer read. Configured Hive credentials
take precedence over legacy OpenAI chat settings. The default Hive generator is
`deepseek-ai/deepseek-v4.1-flash`, configurable through `HIVE_CHAT_MODEL`, with
high reasoning, a 16,384-token completion cap, an 8,192-token thinking budget,
a 120-second request timeout and one provider retry. Provider failures propagate
through the existing SSE/error path without an OpenAI fallback. New evaluation
reports record the actual Hive generator and its request configuration.

A separate live [credential/transport smoke test](../../../../.eval-artifacts/smoke_shared_hive.json)
passed: the production LangChain streaming agent called the deterministic
`issue_read` fixture for comments, consumed the result, and returned a GitHub
citation. The judge returned a valid structured JSON response using the renamed
credential. Three Hive HTTP requests succeeded. No embeddings or application
database writes were involved. This is an integration smoke test, not a new
quality benchmark; historical saved answers and scores remain unchanged.

Verification after this change: **111 offline tests passed**, with the opt-in
database integration test skipped. Ruff, mypy (19 files), formatter checks and
the API compile build passed. New offline regressions cover Hive precedence,
streamed tool-call/result round trips, citation text, HTTP 401 and timeout
propagation, and accurate generator metadata without credentials.

Uncached query embeddings and new indexing still use OpenAI embeddings. A Hive
chat key cannot replace an OpenAI embedding credential; exhausted OpenAI credits
remain a blocker for those calls. Cached benchmark vectors remain usable. No
embedding migration or retrieval-quality tuning was performed in this change.

## Historical maximum scores saved for reporting

At the user's request, [MAXIMUM_SCORES.md](MAXIMUM_SCORES.md) and
[data/maximum-scores-v1.json](data/maximum-scores-v1.json) preserve the selected
historical best readings. Retrieval uses the completed parent-reranker run
(82.35% Recall@6, 46.08% Precision@6). Semantic readings use the earlier valid
partial run (18/24 accepted judgments), with each metric's scored-case count.
This is a per-metric summary across experiments, not a new complete baseline.
Source artifacts, dataset/scorer definitions and current settings are unchanged.

## Local Qwen embeddings and disabled reranking (2026-10-14)

Ollama's `qwen3-embedding:0.6b` now works through the shared production embedding
factory, alongside the explicitly selectable OpenAI provider. Qwen uses native
1,024-dimensional vectors, pinned Q8_0 weights, a repository-retrieval query
instruction and no document instruction. No embedding API key is required.

The original 32,768-token allocation caused partial CPU offloading on the local
6 GB GPU. With an 8,192-token allocation, `ollama ps` reported 100% GPU and about
2.9 GB allocated. Truncation remains disabled; oversized inputs fail explicitly.
Warm query-embedding latency was not isolated, so the roughly seven-second
reranking overhead from the earlier paired run should not be attributed to the
embedding model. No new operational monitoring was added.

The database migration preserved all 674 existing application vectors before
reindexing. The three application repositories were subsequently re-embedded
from their stored snapshots through the production promotion transaction. All
three completed before the user's stop message arrived; file contents and
indexed commit SHAs were preserved. No further re-embedding was performed after
that message. This application migration is separate from evaluation execution,
which still connects only to the dedicated `repomind_eval` database.

Frozen-corpus retrieval runs completed 24/24 cases, with 17 applicable cases:

| Embeddings | Parent reranker | Recall@6 | Precision@6 |
|---|---|---:|---:|
| OpenAI `text-embedding-3-small` | Qwen | 82.35% | 46.08% |
| Local Qwen 0.6B | Qwen | 82.35% | 45.10% |
| Local Qwen 0.6B | Disabled | 77.94% | 37.25% |

The no-reranker run reused the same validated Qwen query vectors. A cache miss
was configured to fail instead of invoking an embedding provider; all cases
completed, proving that this comparison made no embedding calls. It measures
retrieval quality, not isolated warm/cold model latency. Both new runs contain
unscored semantic metrics; the earlier complete judged baseline remains based
on historical saved answers.

At the user's request, the active setting is `REPOSITORY_RERANKER=none`. The
implementation remains available, with a commented `REPOSITORY_RERANKER=qwen3`
setting for future use. Qwen embeddings and Hive inference/judging are retained.
Switching `EMBEDDINGS_PROVIDER` to `openai` requires a matching OpenAI index for
that repository; incompatible vectors are rejected rather than mixed.

Reports: [Qwen with reranker](../../../../.eval-artifacts/retrieval-v9-qwen-embedding-parents.md)
and [Qwen without reranker](../../../../.eval-artifacts/retrieval-v10-qwen-embedding-no-reranker.md).
The [versioned embedding summary](data/qwen-embedding-v1.json) preserves per-case
scores and run identities. Database integration passed separately for both
dimensions with deterministic vectors; these integration vectors are not
quality evidence. Final offline checks passed **133 tests**, with two opt-in
database cases skipped in that run. Ruff, mypy (20 files) and the API compile
build passed.

## Verification

- Offline pytest after identifier/budget changes: **86 passed, 1 integration test skipped**.
- Explicit PostgreSQL integration run: **1 passed**, covering pgvector, corpus
  seeding/isolation, connection serialization, natural-language OR search,
  identifiers, paths, unstemmed tokens, term coverage and stable ranking.
  Vector-only parents report vector provenance; keyword-only succeeds with an
  embedding implementation that rejects query calls. Offline tests also reject
  calls to disabled branches and check mode/budget metadata and batch timeouts.
  New regressions cover ordinary CamelCase prose, explicit/backticked identifiers,
  call syntax, generic-word ranking and both candidate budgets.
- Ruff: **passed**. `mypy app`: **passed**, 18 source files.
- API compile build: **passed**.

These checks are separate from the quality evaluations in the table above.
Deterministic integration embeddings establish mechanics only.
