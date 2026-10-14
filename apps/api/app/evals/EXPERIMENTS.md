# Retrieval improvements and cheaper judge trials

Measured on 2026-10-14. The original [baseline](BASELINE.md) and its GPT-5.4
grades are preserved. Dataset, corpus commit, fixtures, scorer, reference facts,
generation prompt and embedding model are unchanged. These are descriptive
results on the initial benchmark, not a statistical significance claim.
The latest selected configuration has Recall@6 **80.88%** and Precision@6
**38.24%**, using tighter identifier boosts and 24 candidates per branch. Earlier
sections retain the preceding experiments; the identifier/budget section records
the latest changes.

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
