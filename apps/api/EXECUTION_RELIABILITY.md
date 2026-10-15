# Execution reliability pass

The first pass kept the configured models, six-tool/eight-model budgets, 120-second
investigation deadline, hybrid retrieval, reranking setting, frozen benchmark,
fixture matcher, scorer, authorization, and Redis cache-aside design unchanged.

## Version 2 implementation

The next pass keeps production model choices, retrieval/reranking, execution budgets,
authorization, citation catalogs, Redis behavior and native final-answer SSE unchanged.
It changes prompt guidance and MIXED task boundaries, and explicitly versions two
offline evaluation repairs. The earlier captures below remain historical results.

The router's existing call now returns validated, bounded Code and GitHub assignments
for MIXED alongside the existing schedule. Only MIXED uses separate execution
assignments. CODE and GITHUB use the original question directly; only MIXED has
specialist assignments. Each MIXED
specialist receives its assigned subtask as the final user message,
with the original question and history available as context. Sequential Code work
still receives untrusted observed GitHub findings. The synthesizer retains the full
original question. No planning or dedicated final-answer model call is added.

Agent prompts retain synthetic few-shot examples, reinforce completing only the
assigned evidence scope, and tie comment reads to requested live facts. They advise
stopping after a named resource is unavailable, while continuing other assigned
resources. Citation examples now distinguish supporting from unrelated retrieved
windows. Final-generation guidance asks for concise factual sentences with the
smallest sufficient source set and coverage of all assigned parts.

Offline judge v2 partitions exact answer text into sections of at most 600 characters
and batches at most 2400 characters/12 sections. It requires a judgment for each
section, checks quotations within their assigned text, and saves section judgments.
Every section needs claims, limitations or an exclusive nonfactual explanation.
Cited prose cannot be silently classified as nonfactual. Repair attempts retain the
previous labels; global claim IDs are assigned deterministically. Batch concurrency
is bounded at two per case, with cancellation and cleanup on failure. This detects
omitted sections but cannot guarantee complete atomic claim extraction within them.

Two synthetic calibration examples supplement the unchanged original twelve; the
new examples also require minimum claim counts. Models, judge token/time limits and
metric definitions are unchanged. Scorer v2 fixes inline localhost URLs being read
as file citations. Explicit rescoring accepts only the known v1 hash, preserves
previous provenance, and requires a new output path. Reusing calibration requires
matching judge configuration and the complete passing set, with provenance recorded.

Seven supplemental questions and new strict fixtures cover previously untested
helpers, necessary comments, missing-resource comparisons and MIXED assignments.
They were authored after prompt design and before evaluation, against verified source
spans in the same frozen corpus. They are agent-authored supplemental regressions,
not an independent holdout. Original datasets, fixtures and historical reports remain
unchanged. Tests and live evaluations are deferred until this implementation is
complete, as requested.

The initial v2 capture completed 23/24; its seven supplemental cases completed
7/7, but the body-only MIXED case failed one of two additional repeat captures.
Both small prompt ablations (previous few-shot and examples-only, under the same
v2 graph) completed 1/2. These failures were extra comments reads, not unavailable
code evidence or streaming errors. The samples are too small for statistical
claims. All captures are retained under `component-5-quality-v2-*` artifact names.

Trace review found two general problems: task decomposition introduced an
unrequested file-identity goal, and the discussion example still primed comments
reads for a body-only assignment. The final correction asks the router to project
only requested requirements, treats optional unknowns as optional, and supplies a
complete body-only example before the discussion example. GitHub's output-format
example also uses body evidence. No benchmark IDs, expected facts or fixture
responses were added to application prompts.

The initial evaluator passed 14/14 calibration checks. Its obsolete grading batch
was intentionally stopped after a saved checkpoint; it is not a complete semantic
report. The passing calibration is reused with the identical evaluator to avoid
repeating those model calls. Captures and grading use separate artifact names,
preserving the earlier failure counts.

The consolidated API suite reported 367 passed and one new test-fixture failure:
the rescore test incorrectly supplied toy-corpus evidence to the frozen-corpus
validator. Correcting that fixture preserved the validation rule; all 13 evaluator
regressions then passed. After the final prompt correction, all 152 focused agent,
router, graph and native-streaming regressions passed. Ruff, formatting, mypy,
repository build, lock verification and diff checks passed. No intermediate test
execution preceded implementation; the focused reruns addressed failures found by
the consolidated validation.

## Few-shot follow-up and generalization limits

Earlier prompt-only captures completed 23/24; the repeated failure was an optional
comments call absent from a strict frozen fixture. An experimental extra GitHub
intent policy subsequently completed 24/24 but incorrectly blocked a necessary
comment read in a supplemental fact-finding question. That policy and its router
fields have been removed at the user's request. Its execution result does not
establish answer quality. The interrupted grading artifacts are retained.

Production routing has only CODE, GITHUB and MIXED, plus clarification and the
existing MIXED schedule. Agents choose methods from tool descriptions and observed
evidence. GitHub tool descriptions explain details, comments, files, diffs, search
filters, pagination and citation URLs, including the distinction between issue-style
PR conversation and inline reviews. Code tool descriptions explain indexed snapshots,
exact paths, inclusive ranges and stable citations. Evaluation fixtures consume
those same discovered descriptions without changing schemas or matching behavior.
No additional read restrictions or intent-selection abstraction remains.

Offline MCP regressions verify that searches and comments execute with ordinary
request-scoped tools, including after a model-start retry. A separate agent-authored
two-case set checks discussion and fact-finding where requested facts exist only in
a comment. This is supplemental evaluation, not an independent holdout.

The router treats conversation text as untrusted data; it cannot guarantee prompt
injection prevention, including attacks encountered later in retrieved evidence.
Capabilities, request-owned scope, per-call authorization and citation validation
remain enforced in application code. No production judge follows the router;
schema validation and a bounded retry handle malformed classifications. Evidence
scope accuracy and answer quality are assessed offline.

The 24 questions have informed debugging and prompt changes. Treat them as a
development regression set, not a held-out generalization test. High scores on
these questions cannot establish that the agents generalize to unseen repositories
or questions. Few-shot prompting does not by itself remove that risk.

The follow-up replaces the narrow mixed issue/import-loader example and repetitive
GitHub stopping instructions with fictional examples in `app/agents.py`. Three
Code Agent examples cover sufficient evidence, iterative caller inspection, and
argument correction followed by an unavailable file. Five GitHub examples cover
body-only answers, necessary comment reads, continuing after an unavailable resource,
an explicit empty search, and live PR evidence handed to indexed-code investigation.
Positive continuation examples balance stopping examples. Role-specific output
examples match either internal JSON summaries or ordinary marked final answers.

These examples contain no benchmark questions, reference answers or fixture values.
They reside only in the system prompt. They are never inserted as tool messages,
evidence windows or citation entries. Regression tests verify that copied example
sources are rejected as unobserved evidence. No budgets, models, retriever, reranker,
runtime authorization or streaming protocol change in this follow-up.

The prompt version is frozen before its full rerun. A separate, independently
authored holdout with new questions and repositories is still needed to assess
generalization; the fresh 24-case result measures regression behavior only.

The frozen few-shot capture is
`.eval-artifacts/component-5-fewshot-full-v1.json`: **23/24 completed, one strict
execution-evaluation failure**, compared with the previous 24/24 capture. All six
single-file cases, six architecture cases, four GitHub cases and four
insufficient-evidence cases completed; MIXED completed three of four. The graph
returned 17 complete and seven qualified partial workflows, but one partial
workflow is rejected by the strict evaluator. These are distinct counts.

The failed case, `mixed-03`, requested `issue_read(method="get_comments")` while
seeking rationale assigned to indexed-code investigation. The fixture contains no
comments response. The Code Agent completed, and synthesis produced a qualified
answer, but fixture matching correctly remains an evaluation error. The prompt
was not retuned and the fixture was not expanded after observing this failure.

Recall@6 is unchanged at **73.04%**. Precision@6 is **38.24%**, versus 36.27% before;
final evidence coverage remains **100%**, 40/40 units. All three score 17 applicable
cases with seven N/A and zero metric errors, including the observed code evidence
in the failed mixed workflow. These metrics do not imply 24 successful executions.
Deterministic citation validity is **99.78% macro**, 370/371 extracted references;
the unchanged scorer again misidentifies the localhost port as a citation. The
paired deterministic report is `.eval-artifacts/component-5-fewshot-paired-v1.json`.
No answer copied a fictional example path, query or URL, and no missing-marker
failure occurred. One capture does not establish improved reliability or quality.

The existing judge passed **12/12 calibration checks** and graded all **23/23
completed cases**, with zero final judge failures. The failed execution remains
ungraded. The saved report is
`.eval-artifacts/component-5-fewshot-judged-v1.json`; its
`complete_quality_baseline` remains false. The grade command exits nonzero because
the execution failure is preserved, not because grading failed.

| Semantic metric        | Macro result | Scored cases | N/A | Ungraded execution |
| ---------------------- | ------------ | ------------ | --- | ------------------ |
| Groundedness           | 97.32%       | 23           | 0   | 1                  |
| Answer correctness     | 95.24%       | 21           | 2   | 1                  |
| Fact completeness      | 97.62%       | 21           | 2   | 1                  |
| Citation precision     | 81.83%       | 22           | 1   | 1                  |
| Citation coverage      | 76.38%       | 23           | 0   | 1                  |
| Correct abstention     | 100%         | 4            | 19  | 1                  |
| Unnecessary abstention | 0%           | 19           | 4   | 1                  |

Correctness is 20/21 fully correct applicable answers. The omitted reference detail
is the `ACTIVE` user-repository access status in `architecture-01`; no material
factual errors were labeled by the separate correctness task.

The before/after semantic reports have different missing cases: the earlier judge
failed `architecture-03`, while the few-shot capture failed execution of `mixed-03`.
The unchanged comparison command therefore also records common-case deltas in
`.eval-artifacts/component-5-fewshot-semantic-paired-v1.json`: correctness +5.00
percentage points over 20 paired applicable cases, groundedness +0.26 points over
22, fact completeness +0.42 points over 20, citation precision -10.61 points over
21, and citation coverage +4.44 points over 22. These are descriptive development
set results, not evidence of statistical significance or generalization.

Review exposed a judge-quality defect despite calibration and schema acceptance:
the accepted grounding labels for the 553-word `architecture-01` answer contain
only one factual claim after an invalid-quote correction. Numerous other claims
and citations in that answer are omitted. This conflicts with the existing judge
instruction to extract every factual claim and makes its claim-based scores
suspect. The resulting 1/33 citation precision is retained, not silently repaired
or excluded. Other cases also contain noncontributing citation associations, so
the overall citation-precision regression cannot all be attributed to this defect.
Treat the semantic results as provisional pending review of judge extraction.
The judge implementation, calibration set, scorer and accepted labels are unchanged;
no extra judge retry was made after this audit.

The few-shot follow-up passed **353 API tests**, including opt-in PostgreSQL/Redis
integration, plus Ruff, changed-file formatting, mypy (25 source files), repository
build, lockfile verification and diff checks. Six added regression cases cover
both output protocols and rejection of fictional example evidence. Native SSE
streaming tests remain passing. The unchanged web build used the build cache.

## Execution changes

Partial-answer qualification is application-controlled. The stream gate emits a
fixed notice when observed tools establish an unresolved limitation, then streams
the actual final-generation tokens as they arrive. The exact combined text is
validated and persisted. No LLM call is added, and no completed response is split
into artificial deltas. Corrected argument errors remain in the records with a
recovered flag; they do not alone make the answer incomplete. Completed file reads
resolve earlier range limitations. Reading through a known EOF is not truncation.
The context-cache namespace advances to v2 so old EOF warnings are not reused.

Code-tool validation errors describe valid line numbers and the 250-line limit
without echoing arguments. MCP argument-validation errors are bounded and replace
raw validation text with schema guidance. Authorization and scope failures still
stop execution; cancellation is not converted into ordinary missing evidence.

One request-local middleware suppresses unchanged duplicate calls, defers searches
batched with a known-resource read until its details are observed, and removes
tools for the last permitted model turn or repeated unchanged failures. An
unavailable resource does not block investigation of other requested resources.
These controls narrow execution without expanding registry authorization. An agent
can still correct arguments or make a genuinely different investigation request.
Prompts reinforce role boundaries, explicit search selectors, and stopping after
the requested facts are available. Unrelated code diagnosis belongs to Code Agent.

Controlled final-output categories distinguish missing markers, missing completion
signals, truncation, empty answers, streamed-text mismatches, invalid summaries,
unobserved sources, invalid citations, and ignored stopping instructions. They
contain no model response text or stack traces. The marker protocol is unchanged.
The old generic validation failures cannot be retrospectively attributed to a
marker from their saved records; new runs retain specific reasons instead.

Repeated full captures exposed two distinct remaining output failures: abbreviated
source IDs (rejected by citation validation) and missing terminal markers. Numeric
citation instructions were strengthened without relaxing validation. A missing
marker now permits one format correction only when no answer text was emitted and
the original call/time budgets have room. It uses observed tool records, the same
configured model, and actual streamed tokens while retaining the owning MCP scope.
Normal requests have no additional generation. A failed correction or exhausted
budget fails the turn; citations already exposed in a malformed answer are never
silently rewritten or retried.

Timing records distinguish the application notice from native answer tokens, so a
notice alone does not count as the first model-generated answer delta.

A later capture also exposed a model-stream transport failure before any tool or
answer. One application retry is allowed only for that empty start. It reuses the
same owning tool scope and original deadline, charges the failed attempt against
the model-call budget, and reduces the retry's remaining model limit. Discovery,
authorization, tool execution, and partially exposed answers are not replayed.
Repeated transport failure remains FAILED with the controlled `model_transport`
category. Provider SDK retry settings are unchanged.

The final role-specific instructions distinguish numeric indexed-code labels from
GitHub URL citations. A capture exposed an observed GitHub URL accompanied by an
unregistered numeric footnote; the URL itself was valid, but the numeric reference
was correctly rejected. Shared code-citation guidance was removed from the
GitHub-only completion instructions. Explicit file/comment requests can use those
read methods directly; speculative searches still wait for known-resource details.

## Individual fixture audit

All frozen fixtures and matching rules remain unchanged. None of these calls was
silently accepted, remapped to another response, or taught as a gold query string.

| Original case | Unmatched request                                             | Assessment and response                                                                                                                                                                                                                                                                                                                          |
| ------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| missing-03    | `billing` search with open/closed state                       | The question explicitly requested all states. Splitting an already successful empty search changed the requested operation. Preserve explicit selectors and stop with the observed empty result.                                                                                                                                                 |
| missing-04    | PR searches for oversized-file exclusion and file-size limits | Valid API searches, but they cannot establish measurements from the explicitly named unavailable PR. They also mix the Code Agent's assignment into GitHub research. Observe the named PR first and report its unavailability.                                                                                                                   |
| mixed-01      | Source-card searches and broad empty issue/PR searches        | Valid search operations, but unnecessary after the requested issue supplies its live report. Code Agent must investigate indexed source-card behavior. Preserve that role boundary.                                                                                                                                                              |
| mixed-03      | Issue comments and worker/file-size searches                  | The comments call is a legitimate API operation absent from this fixture: an actual fixture-coverage gap for a comments question. This question needs facts already present in issue details plus indexed worker code, so fetching comments is unnecessary. PR/issue searches for indexed implementation are also outside the GitHub assignment. |

The first failed-case rerun completed 7/9. The remaining failures were strict
fixture rejections of a speculative search batched with PR details and an
unnecessary comments read. After the sequencing control and role clarification,
the targeted two-case rerun completed 2/2. These separate runs are not presented as
a fresh 24-case result; the complete evaluation is recorded separately.

## Execution results before the few-shot follow-up

The final capture is `.eval-artifacts/component-5-reliability-full-v6.json`:
**24/24 completed, zero execution failures**, comprising 16 complete and eight
qualified partial workflows. Partial outcomes retain actual errors and limitations;
they are not presented as fully successful investigations. One partial workflow
retains a failed specialist summary while the synthesizer uses its observed tools.

| Category              | Before completed / total | After completed / total | After failures |
| --------------------- | ------------------------ | ----------------------- | -------------- |
| Single-file code      | 5/6                      | 6/6                     | 0              |
| Architecture          | 3/6                      | 6/6                     | 0              |
| GitHub                | 4/4                      | 4/4                     | 0              |
| MIXED                 | 2/4                      | 4/4                     | 0              |
| Insufficient evidence | 1/4                      | 4/4                     | 0              |
| Total                 | 15/24                    | 24/24                   | 0              |

Earlier full captures remain archived: v1 completed 24/24, v2 23/24 (abbreviated
citation), v3 22/24 (two missing markers), v4 23/24 (empty-start transport error),
and v5 23/24 (GitHub numeric footnote). These are different implementation stages,
not repeated samples of a single final version and not a pooled success estimate.

The final unchanged-method retrieval metrics are Recall@6 **73.04%**, Precision@6
**36.27%**, and final evidence coverage **100%** (40/40 required units across all
observed searches/reads). All 17 applicable cases are scored; seven are N/A and
none failed retrieval evaluation. The earlier capture scored only 12 applicable
cases. On those 12 paired cases, Recall@6 is unchanged and Precision@6 is down
2.78 percentage points; the aggregate change is not a controlled retriever change.
The comparison is `.eval-artifacts/component-5-reliability-paired-v6.json`.

## LLM-as-Judge results before the few-shot follow-up

The same existing Hive judge, high reasoning, rubric, schemas, calibration examples,
limits, and scorer were used. The first calibration passed 11/12 and blocked all
grading; its rejected artifact is preserved. One unchanged retry passed 12/12.
That accepted run graded 23/24 answers. `architecture-03` ended with the judge's
output-length limit and has no accepted semantic labels. Approval for an additional
one-case judge retry was denied, so no further model call was made.

The retained report is
`.eval-artifacts/component-5-reliability-judged-v6-retry1.json`. It remains explicitly
**not a complete quality baseline**. Failure counts are retained; unavailable
semantic scores are not replaced with zero or silently presented as measured.

| Metric                 | Macro result                 | Scored cases | N/A | Judge errors |
| ---------------------- | ---------------------------- | ------------ | --- | ------------ |
| Groundedness           | 97.21%                       | 23           | 0   | 1            |
| Answer correctness     | 90.48% (19/21 fully correct) | 21           | 2   | 1            |
| Fact completeness      | 97.22%                       | 21           | 2   | 1            |
| Citation precision     | 93.88%                       | 22           | 1   | 1            |
| Citation coverage      | 71.17%                       | 23           | 0   | 1            |
| Correct abstention     | 100%                         | 4            | 19  | 1            |
| Unnecessary abstention | 0%                           | 19           | 4   | 1            |

Deterministic citation validity is **99.89% macro**, with 416/417 valid extracted
references across 23 applicable cases. The one invalid reference is the unchanged
offline scorer interpreting `//127.0.0.1:8137` as a file-line citation. Its value
is retained; the scorer and benchmark methodology were not changed. The runtime
localhost validator fix is included in the execution capture.

The accepted judge identified two reference-detail omissions: the SSE explanation
omits the user-message response header (`architecture-05`), and the PR comparison
omits that transaction failure can preserve previous rows (`mixed-04`). Both had
the relevant observed evidence. Citation coverage is the largest remaining measured
quality gap. These results support improving answer completeness and citation
attachment before changing retrieval, reranking, or models.

## Verification

- API suite with opt-in database/Redis integration: **347 passed**.
- Native SDK streaming tests cover ordinary direct generation, application notices,
  corrected arguments, malformed-marker correction, exhausted budgets, repeated
  connection failure, no replay after exposed text, and GitHub MCP lifetime.
- Ruff, changed-file formatting, mypy (25 source files), repository build, lock
  verification, and diff whitespace checks passed. The unchanged web build was
  served from the existing build cache.
- Frozen datasets, fixtures, corpus references, scorer, judge implementation,
  retrieval/reranking code, dependency locks, model choices, and agent limits are
  unchanged. No production judge or legacy fallback was added.

## Final simplified-router capture

The current application is captured in
`.eval-artifacts/component-5-quality-v2-simple-full.json`. Execution completed
**23/24 cases**, retaining one failure; all 24 router scopes matched the frozen
annotations. The annotations were never passed to the router. All workflows
returned, with 19 complete and five qualified partial results; the strict fixture
check rejects one of those partial results. This is not complete quality validation.

| Category              | Completed / total | Execution failures |
| --------------------- | ----------------- | ------------------ |
| Single-file code      | 6/6               | 0                  |
| Architecture          | 6/6               | 0                  |
| GitHub                | 4/4               | 0                  |
| MIXED                 | 3/4               | 1                  |
| Insufficient evidence | 4/4               | 0                  |
| Total                 | 23/24             | 1                  |

`mixed-03` successfully read issue details, then unnecessarily requested comments
to discover a filename/path not required by the question. The existing fixture
does not cover that read. It remains a strict fixture failure; no fixture, expected
fact, budget or model was changed to make it pass. Prompt guidance and descriptions
have not eliminated this exploration behavior reliably.

The seven supplemental cases completed 7/7, two representative repeats completed
2/2, and the two additional MIXED discussion/fact-finding cases completed 2/2.
Both additional cases executed real fixture-backed `get_comments` calls and used
their observed results. These agent-authored sets are regressions, not an
independent holdout or a statistical reliability estimate.

## Final offline judge results

Judge v2 retained its passing 14/14 calibration with exact configuration matching.
The saved few-shot answers were rescored and regraded with this evaluator: 23/24
accepted grades, one execution failure ungraded. Twenty-one accepted historical
grades were reused with their provenance; the remaining two were completed without
rerunning their investigations. On the same saved 553-word architecture answer,
v1 extracted one claim while v2 extracted 35 across twelve sections.

The new capture received **22/24 accepted grades**: one execution failure remains
ungraded, and one completed answer failed offline judge validation. Those two
cases are not silently counted as successes or assigned invented semantic scores.
The reports are `component-5-quality-v2-simple-before.json`,
`component-5-quality-v2-simple-after.json`, and
`component-5-quality-v2-simple-comparison.json` under `.eval-artifacts`.

| Metric                  | Before macro | After macro | Before scored / N/A / unscored / errors | After scored / N/A / unscored / errors |
| ----------------------- | ------------ | ----------- | --------------------------------------- | -------------------------------------- |
| Recall@6                | 73.04%       | 81.86%      | 17 / 7 / 0 / 0                          | 17 / 7 / 0 / 0                         |
| Precision@6             | 38.24%       | 41.18%      | 17 / 7 / 0 / 0                          | 17 / 7 / 0 / 0                         |
| Final evidence coverage | 100%         | 95.59%      | 17 / 7 / 0 / 0                          | 17 / 7 / 0 / 0                         |
| Groundedness            | 96.00%       | 99.50%      | 23 / 0 / 1 / 0                          | 22 / 0 / 1 / 1                         |
| Answer correctness      | 95.24%       | 90.00%      | 21 / 2 / 1 / 0                          | 20 / 2 / 1 / 1                         |
| Fact completeness       | 97.62%       | 95.83%      | 21 / 2 / 1 / 0                          | 20 / 2 / 1 / 1                         |
| Citation precision      | 85.14%       | 84.32%      | 22 / 1 / 1 / 0                          | 21 / 1 / 1 / 1                         |
| Citation coverage       | 71.45%       | 77.41%      | 23 / 0 / 1 / 0                          | 22 / 0 / 1 / 1                         |
| Citation validity       | 100%         | 100%        | 23 / 1 / 0 / 0                          | 23 / 1 / 0 / 0                         |

Unscored semantic cases are execution failures; semantic errors are judge failures.
Retrieval remains defined on the first search's six contexts. Final coverage is
separate: 38/40 required evidence units were observed, compared with 40/40 before.
Deterministic retrieval/citation metrics can still score observed evidence in the
strict fixture failure; they do not change that case's failed execution status.
All 274 extracted references in the new capture were valid. Correct abstention
was 100% on four applicable judged cases; unnecessary abstention was 0% on eighteen.
Both abstention metrics retain one execution and one judge failure outside their
measured values.

Restricting the semantic comparison to cases scored in both reports gives:

| Paired metric      | Cases | Before | After  |
| ------------------ | ----- | ------ | ------ |
| Groundedness       | 22    | 95.82% | 99.50% |
| Answer correctness | 20    | 95.00% | 90.00% |
| Fact completeness  | 20    | 97.50% | 95.83% |
| Citation precision | 21    | 85.68% | 84.32% |
| Citation coverage  | 22    | 72.10% | 77.41% |

These results are mixed: grounding and citation coverage rose, while completeness,
correctness, citation precision and final evidence coverage fell. This is not a
demonstrated overall quality improvement, and the failures prevent a complete
quality baseline. Repeated use of the development benchmark also prevents claims
of held-out generalization.

The supplemental judge accepted 7/7 answers: groundedness 98.98%, correctness and
completeness 100% on six applicable cases, citation precision 93.15%, and citation
coverage 87.11%. Both additional discussion answers were accepted and fully correct;
citation precision was 94.12% and coverage 86.51%. These easier, agent-authored cases
cannot substitute for an independent holdout.

## Remaining diagnoses

- `architecture-05` failed offline validation because section splitting isolated
  `[7][6].` into a citation-only section. The model labeled it nonfactual formatting,
  but the validator rejected any cited section without claims or limitations.
  All bounded repair attempts failed. This evaluator false positive must be repaired
  before treating that answer as graded; the current run does not weaken the rule
  or substitute a score. The captured answer also ends with an unwanted
  `</final_answer>` control tag. That is a separate formatting defect, not the
  reason the judge rejected it, and not a failure of native token streaming.
- Both versions omit the active user-repository relationship detail in the
  repository-connection trace. The new source-card explanation also omits an
  explicit distinction between a displayed source and support for individual
  claims. No gold facts were inserted into application prompts to repair these
  benchmark-specific omissions.
- Optional GitHub exploration still causes the one strict execution failure.
  An intent classifier is not a reliable remedy: its removed experimental policy
  blocked a legitimately necessary comment read. Further work should assess
  selection behavior on independently authored questions, alongside the evaluator
  boundary defect, before another round of prompt tuning or retrieval changes.

Observed model calls fell from 111 to 104 across the 24 workflows. Input tokens
fell from 1,436,808 to 1,210,825; output tokens fell from 38,885 to 27,951. Mean
workflow duration rose from 17.65 to 19.17 seconds. These are single-capture
observations, not controlled latency or cost guarantees; no price estimate is made.

Final verification: **372 API tests passed**, including all five opt-in integration
tests. Ruff, changed-Python formatting, mypy (25 source files), lock verification,
repository build and whitespace checks passed. The unchanged web build used its
cache. Frozen benchmark, fixtures, original calibration, retriever, reranker and
dependency locks are unchanged. No production judge, extra router intents, legacy
fallback or additional specialist final-answer call was added.

## Reliability and completeness pass, evaluator v3 (2026-10-16)

The final capture completed **24/24 executions and 24/24 accepted judge grades**.
It meets the requested single-capture correctness, completeness and citation-validity
thresholds. **The overall acceptance gate is not met:** targeted repetitions
completed 14/15, and investigation status/provenance defects remain. This is not a
claim that all failures are fixed or that the system generalizes beyond these
development regressions.

### Changes

- Accept citation-only punctuation sections as formatting in the offline judge,
  while continuing to reject cited factual prose labeled nonfactual. Citation
  masking uses original offsets, including overlapping grouped references. Exact
  quotations, evidence validation and full section accounting remain enforced.
  Judge configuration v3 adds one separate synthetic calibration example; original
  and v2 calibration files are unchanged.
- Remove an optional terminal `</final_answer>` across native token boundaries.
  Hold only a possible protocol suffix, stream ordinary content immediately, and
  preserve literal interior marker examples. Streamed, validated and persisted
  text must agree. Direct specialists still use their existing final generation;
  no extra answer call or simulated streaming was introduced.
- Keep the original question as the final user message during MIXED investigation.
  Present generated router assignments as scope suggestions that cannot add user
  requirements. Short shared guidance checks requested parts, relevant conditions
  and effects, supporting citations, and when further tool calls are useful.
  Fictional few-shot examples distinguish a live report handed to the Code Agent
  from a request that actually needs discussion comments. They contain no benchmark
  answers, paths or gold values.
- Distinguish optional unread `context_notes` from incomplete requested ranges in
  code tools. Cache the distinction under Redis namespace v3; old entries expire
  without a flush. Corrected/follow-up reads can resolve the original requested
  interval without requiring the rest of the file. Preserve an observed unresolved
  read gap even if an LLM summary omits it.
- Prompt both specialists and synthesis to compare live proposals with indexed
  behavior as different observed versions. Evaluation still exposes cases where
  this guidance fails to prevent false missing-information/status reports.

Authorization, scoped MCP ownership, database pooling, embedding behavior, models,
agent limits, retriever, reranker, scorers and fixtures are unchanged. Judging remains
offline. The existing [evaluation guidance](https://docs.langchain.com/langsmith/evaluation-types)
distinguishes benchmark comparisons, regression checks and deterministic validators;
those results are reported separately here.

### Captures and execution reliability

All final artifacts use prefix `component-5-reliability-v3-scoped` under
`.eval-artifacts/`:

- `-pipeline.json`: every completed capture and grading outcome.
- `-full.json`: the single frozen 24-case execution capture.
- `-before.json`: saved `component-5-quality-v2-simple-full.json` answers, freshly
  graded with v3; no v2 grades reused.
- `-after.json`: new benchmark answers graded with the same v3 configuration.
- `-repeat-{1..5}-{mixed,comments}.json`: all five repetitions of each selected case.
- `-supplemental.json` and `-supplemental-judged.json`: existing supplemental seven.
- `-comments-judged.json`: the preselected first repetition of both necessary-comment
  cases; not the best repetition chosen afterward.
- `-comparison.json` and `-analysis.json`: paired metrics and execution trajectories.

| Category              | Before execution | New execution | New execution failures |
| --------------------- | ---------------- | ------------- | ---------------------- |
| Single-file code      | 6/6              | 6/6           | 0                      |
| Architecture          | 6/6              | 6/6           | 0                      |
| GitHub                | 4/4              | 4/4           | 0                      |
| MIXED                 | 3/4              | 4/4           | 0                      |
| Insufficient evidence | 4/4              | 4/4           | 0                      |
| Total                 | 23/24            | 24/24         | 0                      |

All 24 new router scopes match the unchanged offline annotations; annotations were
not passed to the router. The new workflow reports 18 complete and six qualified
incomplete results, versus 19 and five previously. Execution completion means a
returned, validated answer, including a useful partial answer; it does not mean
every specialist succeeded or every question was fully supported.

| Targeted regression        | Execution completion | Failures |
| -------------------------- | -------------------- | -------- |
| `mixed-03`                 | 4/5                  | 1        |
| `method-mixed-discussion`   | 5/5                  | 0        |
| `method-mixed-fact-finding` | 5/5                  | 0        |
| Total                      | 14/15                | 1        |

In MIXED repetition 2, the GitHub Agent read issue details successfully, then
unnecessarily requested `issue_read(method="get_comments")` for issue 103. The
strict frozen fixture does not cover this call. It is a valid API operation but
unnecessary to explain the reported byte size using indexed worker code, so the
failure is retained rather than extending the fixture. The other four repetitions
and the full capture used only `get` for that issue. Both necessary-comment cases
executed successful fixture-backed `get_comments` calls in **every** repetition.
Their ten executions returned answers but were all marked incomplete, exposing
the separate scope/limitation problem. All 176 extracted references across the
15 repetitions were valid, including those in the failed execution's partial answer.

Earlier attempts with prefixes `component-5-reliability-v3` and
`component-5-reliability-v3-final` were stopped during targeted repetitions to
correct scope guidance before the final capture. Each has four case records in
checkpointed captures: one failed MIXED repetition, one successful MIXED repetition, and two
successful comment cases. The second also has one completed discussion case in a
partially written next capture; the first's next capture has no case records.
Those artifacts and failures remain intact. They use earlier application hashes,
are not pooled with the final repetitions, and are not hidden as successful trials.

### Fresh offline judging and metrics

Fresh v3 calibration passed **15/15**. Baseline and new judge configuration objects
match exactly, including rubric, implementation, schemas, calibration hashes,
provider, model, limits and reasoning effort. Only this new calibration was reused
for subsequent grading, with its source report hash recorded.

The new benchmark has **24/24 accepted grades**, with no unscored or errored semantic
metrics. `mixed-04` needed one bounded exact-quote repair, which succeeded. The saved
baseline has **22/24 accepted grades**: its original `mixed-03` execution failure
remains ungraded, and `architecture-05` still exhausts cited-prose validation. Its
isolated `[7][6].` section now satisfies the deterministic formatting rule; this
remaining error means another cited section was labeled nonfactual. Invalid model
labels are not retained with a section ID, so the precise offending section cannot
be established from the saved report. The historical answer, including its leaked
closing tag, was not rewritten to make it pass. No synthetic score fills either gap.

All values below are case macro averages, not pooled claim/fact ratios. S/N/U/E means
scored / not applicable / unscored execution failures / judge errors. Deterministic
evidence and citation metrics can score observed evidence in an execution failure;
that does not change its failed execution status.

| Metric                  | Saved baseline, v3 | New answers, v3 | Before S/N/U/E | After S/N/U/E |
| ----------------------- | ------------------ | --------------- | -------------- | ------------- |
| Recall@6                | 81.86%             | 77.45%          | 17/7/0/0       | 17/7/0/0      |
| Precision@6             | 41.18%             | 38.24%          | 17/7/0/0       | 17/7/0/0      |
| Final evidence coverage | 95.59%             | 100%            | 17/7/0/0       | 17/7/0/0      |
| Groundedness            | 96.45%             | 97.80%          | 22/0/1/1       | 24/0/0/0      |
| Answer correctness      | 95.00%             | 95.45%          | 20/2/1/1       | 22/2/0/0      |
| Fact completeness       | 97.50%             | 97.73%          | 20/2/1/1       | 22/2/0/0      |
| Citation precision      | 81.86%             | 90.19%          | 21/1/1/1       | 23/1/0/0      |
| Citation coverage       | 75.27%             | 74.20%          | 22/0/1/1       | 24/0/0/0      |
| Citation validity       | 100%               | 100%            | 23/1/0/0       | 23/1/0/0      |
| Correct abstention      | 100%               | 100%            | 4/18/1/1       | 4/20/0/0      |
| Unnecessary abstention  | 0%                 | 0%              | 18/4/1/1       | 20/4/0/0      |

New correctness is 21/22 fully correct applicable answers. Completeness covers
42/43 reference facts; its reported 97.73% is the mean per-case fraction. All
334/334 extracted references are valid, versus 274/274 previously. Final code
evidence covers 40/40 required units versus 38/40; it is explicitly separate from
Recall@6. Recall@6 and Precision@6 still use the first search's six contexts.

Restricting comparison to identical scored cases avoids attributing restored judge
coverage to answer-quality improvement:

| Paired metric      | Cases | Before | After  |
| ------------------ | ----- | ------ | ------ |
| Groundedness       | 22    | 96.45% | 97.72% |
| Answer correctness | 20    | 95.00% | 95.00% |
| Fact completeness  | 20    | 97.50% | 97.50% |
| Citation precision | 21    | 81.86% | 90.08% |
| Citation coverage  | 22    | 75.27% | 74.68% |

The observed improvements are execution/judge coverage, final evidence coverage,
grounding and citation precision. Paired correctness and completeness are unchanged.
First-pass retrieval and citation coverage regress, so this is not an improvement
in every metric. Lower first-pass Recall@6 comes from the repository-connection
case's changed initial search; later retrieval recovers its required evidence.
Additional file reads recover the previous final-evidence gaps in architecture
cases 3 and 4. No retriever/reranker or gold annotation was changed.

The supplemental set completed and graded **7/7**: correctness/completeness 100%
on six applicable cases, groundedness 100%, citation precision 89.47%, citation
coverage 88.95%, citation validity 100% (68/68 references), Recall@6 100% and
Precision@6 25% on four applicable cases, final evidence coverage 100%.
The two preselected necessary-comment answers graded **2/2**, with correctness and
completeness 100%, groundedness 97.56%, citation precision 98.48%, citation coverage
86.25%, citation validity 100%, Recall@6 100% and Precision@6 16.67%.
One exact limitation quote needed a successful bounded repair. These authored
regressions are not an independent holdout.

### Remaining diagnoses and acceptance

- The zero-failure repeated-execution gate fails because of the extra comment read.
  Prompt changes improve observed selection but do not enforce minimal sufficient
  investigation. Necessary comment reads still work; globally blocking comments
  would conceal this defect and break legitimate questions.
- Agents still sometimes call a fact belonging to the other specialist missing,
  or say a live PR cannot be compared because it is absent from the indexed commit.
  Synthesis can answer from both catalogs, but graph status still aggregates those
  model-authored limitations. This causes false qualified partials even in otherwise
  correct answers. Prompt guidance alone has not repaired status semantics reliably.
- The new `mixed-02` answer completes from observed evidence, but its GitHub
  investigation has a controlled `summary_schema` validation error after the
  fixture's 503 and a blocked repeated request. The graph retains that error and
  returns a qualified partial; 24/24 execution must not hide this specialist failure.
  The 503s in MIXED PR cases and the unavailable PR in `missing-04` remain real
  observed limitations, independently of spurious provenance notes.
- `architecture-01` still omits the active status of the user-repository link despite
  observing the source. This is the one missing reference fact and the only
  applicable answer failing the all-required-facts correctness criterion. Do not
  insert that benchmark-specific fact into production prompts.
- Citation coverage remains the largest measured answer-quality gap: some supported
  claims lack their own supporting reference or attach the wrong window. Valid
  source IDs alone do not prove support. Unsupported additions also remain, such as
  universal commit-pinning despite a branch fallback. The judge additionally flags
  incomplete section fragments and reasonable library-semantic inferences (e.g.
  transaction rollback and the MiB conversion); these are retained for review rather
  than post-hoc relabeling.
- The historical judge failure remains, although the citation-only regression and
  calibration pass. More precise saved rejected-section diagnostics are needed to
  identify its remaining model misclassification without relaxing factual-prose
  validation.

The requested 24/24 execution, 24/24 new grades, correctness >=95%, completeness
>=97.5%, and citation validity 100% gates pass for the final single capture. The
zero-failure repetition gate does not. Models, budgets and strict fixtures were not
changed to satisfy a gate. Keep these failures as development regressions; an
independently authored holdout is still required before claiming generalization.

### Latency and usage

These are recorded production-workflow observations across all 24 cases, including
the failed baseline execution; judge calls are excluded. Offline judge token usage
is not recorded by the existing runner. Single captures do not establish a causal
latency or cost improvement.

| Measurement                    | Before    | After     |
| ------------------------------ | --------- | --------- |
| Mean workflow seconds          | 19.17     | 22.54     |
| Median workflow seconds        | 18.45     | 22.02     |
| Maximum workflow seconds       | 35.20     | 50.89     |
| Mean seconds to first delta    | 16.45     | 18.23     |
| Median seconds to first delta  | 15.23     | 16.04     |
| Completed model calls          | 104       | 110       |
| Tool calls                     | 76        | 78        |
| Repository searches            | 49        | 46        |
| File reads                     | 4         | 11        |
| Input tokens                   | 1,210,825 | 1,249,142 |
| Output tokens                  | 27,951    | 40,107    |

The additional reads improve final evidence coverage but do not guarantee complete
answers. Redis hit behavior is verified by regressions/integration tests; these
offline benchmark captures do not measure a Redis cache speedup. No closing marker
leaked in the new 24-case or supplemental capture, versus 18 historical answers.

### Verification actually run

- Full API suite with all opt-in integrations: **401 passed, 1 failed** out of 402.
  The new explicit-read regression exposed a real summary-erases-observed-gap bug.
  After fixing it, the affected agent/graph/native-streaming tests passed **123/123**.
- After correcting original-question authority, affected agent/graph/streaming/router
  tests passed **181/181**. The final few-shot string change passed **6/6** focused
  protocol tests. Thus all 402 distinct tests passed across the full run and focused
  reruns; there was no fresh single full-suite `402 passed` output after the fix.
- Final Ruff on `app tests`, formatting checks on all nine changed Python files,
  mypy (25 source files), `uv_lock.py --check` (151 packages), `pnpm build`, and
  diff whitespace checks passed. The unchanged web build used its existing cache.
- Frozen benchmark, fixtures, supplemental/method datasets, original/v2 calibration,
  scorer, retrieval/reranking and dependency locks have no diff. Tracked generated
  bytecode was restored after the build. No frontend or migration was changed.
