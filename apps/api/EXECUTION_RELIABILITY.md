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

## Targeted follow-up (2026-10-16)

This follow-up ran **12 selected executions across eight distinct questions**, not
another full benchmark. All 12 executions completed: **11 complete answers and one
qualified partial** retaining the fixture's HTTP 503. The three selected citation
answers received **3/3 accepted grades**. These results do not replace the previous
24-case measurements or establish the original five-repetitions-per-case gate.

### Changes and focused checks

- Add optional `context_notes` to specialist summaries, investigation findings and
  workflow results. Handoffs, provenance differences and optional context have a
  separate output channel from unresolved requested facts. Notes are retained for
  synthesis and evaluation without triggering a partial notice. Actual missing
  information and unrecovered execution errors still qualify an answer; authorization
  and scope failures still stop it. No free-text limitation is silently deleted.
- Remove synthetic "agent is incomplete" entries from combined missing-information
  lists; individual statuses remain available in findings. The graph determines
  qualification from the actual gaps and errors rather than duplicating status as
  a new factual limitation. Unknown semantic gaps still require truthful model
  reporting; the application does not implement a production grader.
- Retain at most five schema-owned field names/list positions and validation error
  categories for invalid summaries. Exclude validation inputs, raw output, unknown
  field names, context and stack traces. Existing required fields and strict
  validation remain unchanged; older three-field summaries default to empty notes.
- Add a small GitHub checkpoint after observed resource details: further reads need
  an unanswered live fact from the original request. Enrich issue-tool descriptions
  with the distinction between discussion and indexed implementation. Comments remain
  callable; no intent classifier, selection model, hard block or permissive fixture
  was introduced.
- Add one shared citation rule covering factual summaries/conclusions and statements
  combining a value with behavior from another window. No benchmark facts or new
  few-shot examples were inserted. Models, budgets and retrieval remain unchanged.

The affected agent, parent-graph and native-SSE batch passed **135/135 tests**,
including 12 added cases for notes versus genuine gaps and redacted/bounded schema
diagnostics. Mypy initially found a dictionary-default inference problem in the new
checkpoint; explicit observed-call narrowing fixed it. Its four affected handoff
cases then passed **4/4**. Changed-file Ruff, formatting (seven Python files), mypy
(four changed application modules) and diff whitespace checks passed. No full API
suite, opt-in integration suite, web build or lock verification was repeated for
this change. The targeted live runs exercised the existing isolated evaluation DB.

### Live execution results

Artifacts use prefix `.eval-artifacts/component-5-targeted-v4`:
`-pipeline.json`, `-summary.json`, `-issue-status.json`, `-pr-status.json`,
`-mixed-repeat-{1..5}.json`, `-comments.json`, and `-citations.json`.
All captures use the same application hash; every capture and failure is retained.

| Check | Result | Observed behavior |
| ----- | ------ | ----------------- |
| Unstable `mixed-03` | 5/5 complete | Only `issue_read(get)`; no unnecessary comments or fixture misses. Previous repetitions completed 4/5. |
| `mixed-01` issue/code status | Complete | Previously partial due to another role's scope; notes retained, no missing facts or execution errors. |
| `supp-mixed-02` PR/code status | Complete | Different live/indexed versions remain provenance notes; actual PR details/files/diff and indexed evidence retained. |
| Both necessary-comment cases | 2/2 complete | Each executed successful fixture-backed `get_comments`; previously marked incomplete. One execution each, not five. |
| `mixed-02` summary diagnostic | Qualified partial | No summary-schema failure on this attempt; actual `get_diff` 503 remains unrecovered. A duplicate request was blocked and recorded as recovered. |
| `code-03`, `code-04` citations | 2/2 complete | Actual final specialist generations, with observed code evidence and validated references. |

There were zero failed executions or unmatched fixtures, but the genuine tool
failure was not counted as success of every specialist. All **158/158 extracted
references** were valid. The intermittent historical `summary_schema` failure was
not reproduced in its one targeted run; this does not prove it is eliminated or
identify its original cause. New diagnostics make a future occurrence actionable.

### Three-answer judge comparison

The three fixed cases are `mixed-01`, `code-03`, and `code-04`. Reports are
`-citation-before.json`, `-citation-after.json`, and `-citation-comparison.json`.
The baseline subset preserves existing accepted v3 grades from
`component-5-reliability-v3-scoped-after.json`; it was not regenerated or rejudged.
The unchanged v3 judge configuration matches exactly. Its passing 15-case calibration
was reused with recorded provenance, not rerun. All three new grades were accepted
without a validation repair; every metric below has three scored cases and no
N/A, unscored or errored cases.

| Case-macro metric | Saved matched subset | New matched subset |
| ----------------- | -------------------- | ------------------ |
| Answer correctness | 100% | 100% |
| Fact completeness | 100% | 100% |
| Groundedness | 100% | 98.77% |
| Citation precision | 85.48% | 89.71% |
| Citation coverage | 57.99% | 67.75% |
| Citation validity | 100% | 100% |
| Recall@6 | 88.89% | 55.56% |
| Precision@6 | 27.78% | 27.78% |
| Final evidence coverage | 100% | 100% |

Citation coverage improves in all three: `mixed-01` 60.87% to 74.07%, `code-03`
52.00% to 66.67%, `code-04` 61.11% to 62.50%. Coverage is still incomplete. Some
factual introductions remain uncited, and some numeric/behavior claims still omit
one of their required windows. The judge also retains section-boundary fragments.
The one unsupported new claim says the system prompt "only" contains a particular
citation instruction, although the observed prompt contains additional instructions.
No rejected claim was relabeled to improve metrics.

Recall@6 declines because the first `mixed-01` search did not recover its complete
annotated evidence unit; subsequent searches/reads did. The other two cases' Recall@6
is unchanged. All five required units were recovered across the final investigations.
This is iterative evidence coverage, not Recall@6. No retriever, scorer, dataset,
fixture, calibration, dependency lock or Redis behavior was changed.

On these same three cases, mean workflow latency rose **24.59s to 29.80s**, completed
model calls **13 to 15**, input tokens **160,833 to 197,224**, and output tokens
**4,997 to 7,100**. Across all 12 targeted executions, mean latency was 30.87s,
with 74 completed model calls, 966,222 input tokens and 30,296 output tokens.
These counts exclude offline judge usage, which the runner does not record. They
are observations of a small subset, not causal performance or generalization claims.

The selected reliability regressions pass, with truthful partial handling preserved.
Remaining limitations include incomplete citation attachment, the observed retrieval,
grounding and latency regressions, the unreproduced intermittent schema failure,
and unknown behavior on untested cases. The historical repository-link omission and
historical judge rejection were not rechecked here. No further prompt/evaluation loop
was run; a full acceptance checkpoint and independent holdout remain separate work.

## First-search recall and latency pass (2026-10-16)

This pass executed `mixed-01` three times and graded **trial 1 only** with the
unchanged v3 judge. All three executions completed with complete workflow status,
no errors or requested-evidence gaps, and **49/49 valid extracted references**.
The judge accepted trial 1 without validation repairs and matched all three
reference facts. The all-three first-search recovery target **failed**: trial 3
needed a file read. No additional tuning or evaluation loop followed that result.
These are three executions of one development regression, not a new benchmark
score or evidence of generalization.

### Implementation and focused checks

- `AgentFindings.to_context_payload()` is shared by synthesis and sequential
  GitHub-to-Code handoff. It keeps canonical evidence intact and removes only a
  successful tool source's repeated `content` whose ID and text match canonical
  evidence exactly. Unmatched text and failed results stay intact. Source
  associations, call IDs, paths/ranges, URLs, arguments, status, errors and
  limitations remain in context. Original findings, native messages and artifacts
  are unchanged. Both consumers explain ID-based canonical evidence lookup.
- Only the code-search tool description gains a query-fidelity instruction:
  preserve the requested behavior and explicit identifiers/paths. No specialist
  prompt examples, models, reasoning settings, budgets, routing strategies,
  retrieval/reranking parameters, SSE protocol or cache behavior changed.
- Offline callbacks add `execution.call_timings` keyed by native run ID, with
  parent ID, role, start offset, duration and success/error/canceled outcome.
  Tool names are included; message bodies, tool arguments, outputs and exception
  bodies are not. In-flight callbacks are finalized when the workflow unwinds.
  Existing provider usage, model counts and first-answer-delta timing remain.
- The focused batch passed **145/145 tests** (62 unrelated evaluation tests
  deselected): agents, parent graph, native SSE and eight evaluation-timing cases.
  This includes two new lossless-projection regressions and real graph callback
  coverage. Changed-file Ruff and formatting passed for seven Python files;
  mypy passed for four changed application modules. Initial lint line lengths and
  a callback-metadata typing issue were corrected before the live captures.
  No full suite, opt-in integration suite, full benchmark, build, lock check,
  recalibration or repeat of the saved 17-query retrieval investigation ran.

The exact test selection from `apps/api` was:

```powershell
python scripts/uv_run.py pytest -p no:cacheprovider tests/test_agents.py tests/test_repository_graph.py tests/test_chat_streaming.py tests/test_evals.py -k 'not test_evals or execution_timing'
```

### Payload reduction

Using saved slow-case evidence and tool records from
`component-5-targeted-v4-citation-after.json`, equivalent synthesis payloads before
and after projection measured **160,132 to 86,368 characters (46.06% smaller)**,
or 160,144 to 86,378 UTF-8 bytes. All 11 canonical code windows (55,896 text
characters) remain, and all 14 omitted copies resolve to exactly their original
text. This exceeds the 30% target without truncation or summarization.

This is a reconstruction from saved complete evidence, tool records, catalogs and
agent summaries, not a capture of the original prompt bytes. Unsaved per-agent
`source_ids` and `model_calls` use identical defaults in both versions. The
measurement and resolution assertions are saved in `-projection.json`.

### Three live trials

Artifacts use `.eval-artifacts/component-5-context-v5`: `-trial-{1..3}.json`,
`-trial-1-judged.json`, `-saved-before.json`, `-trial-1-comparison.json`,
`-projection.json`, `-pipeline.json`, and `-summary.json`. All three captures have
the same application hash. Dataset, strict fixtures, corpus, scorer, judge
configuration and the saved baseline report are unchanged.

| Measurement | Saved slow run | Trial 1 | Trial 2 | Trial 3 |
| ----------- | -------------- | ------- | ------- | ------- |
| Execution / workflow | Completed / complete | Completed / complete | Completed / complete | Completed / complete |
| Recall@6 | 0% | 100% | 100% | **0%** |
| Precision@6 | 0% | 16.67% | 16.67% | 0% |
| Final evidence coverage | 100% | 100% | 100% | 100% |
| Citation validity | 20/20 | 14/14 | 16/16 | 19/19 |
| Workflow seconds | 57.906 | 53.609 | 36.703 | 44.562 |
| First answer delta seconds | 54.828 | 52.328 | 32.688 | 42.500 |
| Completed model calls, including router | 8 | 6 | 7 | 8 |
| Tool calls | 5 | 3 | 4 | 5 |
| Input tokens | 129,615 | 61,646 | 63,190 | 100,172 |
| Output tokens, including reasoning | 4,445 | 5,182 | 4,323 | 5,484 |
| Reasoning tokens | 2,744 | 3,761 | 2,610 | 3,675 |
| Synthesis input tokens | 46,877 | 25,854 | 23,486 | 27,632 |

Recorded first queries, in trial order:

1. `source cards chat retrieved sources display answer citations`
2. `chat source cards retrieved candidates citations answer claims`
3. `streaming source cards retrieved candidates citations chat`

Each trial selected `GITHUB_THEN_CODE` and made two concurrent search calls in its
first Code Agent step. The recorded first query is the first search to enter the
existing code-tool method; result-record order can differ during concurrency.
Recall@6 still scores that single search's six windows, not the union of searches.

- Trial 1: issue `get`, then the first query above alongside
  `chat streaming source card citation rendering`. No file read or comment read.
- Trial 2: issue `get`, then `get_comments`, then the first query above alongside
  `streaming source card display chat answer cite`. No file read.
- Trial 3: issue `get`, then `get_comments`, then the first query above alongside
  `chat message source card attach`, then `get_file_context` for
  `apps/api/app/chat.py:350-470` (observed window 350-446).

Every tool call matched the strict fixtures or returned actual indexed evidence.
The extra comment reads in trials 2 and 3 repeated facts already present in the
issue body; they add work without being necessary for the requested report.

### Stage timing and remaining failures

Observed role spans below run from the role's earliest native callback start to
its latest callback end. They include gaps within the role and overlapping tool
calls only once. They are not summed durations or a breakdown of database versus
embedding time. The historical run has no such callback measurements.

| Role span, seconds | Trial 1 | Trial 2 | Trial 3 |
| ------------------ | ------- | ------- | ------- |
| Router | 4.469 | 2.547 | 1.625 |
| GitHub investigation | 9.110 | 8.703 | 9.235 |
| Code investigation | 19.437 | 10.407 | 12.422 |
| Synthesis model | 20.062 | 14.906 | 21.156 |

Median workflow latency is **44.562s**, 23.04% below the saved slow run; median
first delta is **42.500s**, 22.48% lower. Median input tokens fall **51.25%** to
63,190, and synthesis input tokens fall **44.85%** to 25,854. Median completed
model calls fall from 8 to 7. These observations compare three new runs to one
saved slow run; provider caching and runtime variance prevent a causal claim.
The first trial's two search durations were 7.61s and 7.77s and overlapped; later
searches took 0.72-0.97s. Timing does not identify which retrieval substage caused
that difference. Redis is not enabled in this existing offline runner.

Median output tokens rise **16.58%** to 5,182, and reasoning tokens rise **33.93%**
to 3,675. Synthesis still takes 14.91-21.16s; its median reasoning count rises
from 1,967 to 3,076 despite fewer input tokens. Reasoning settings were preserved.

Trial 3's first six windows contain `chat.py:129-376` but omit the required later
backend citation loop. First-pass evidence overlap is only 20%; the second query
also fails to establish the full requested backend behavior. The later exact file
read recovers it. This is consistent with the previously observed parent-window
selection sensitivity, but this pass did not rerun candidate-pool analysis or
change ranking. The tested full-pool sorting alternative remains rejected after
its saved 17-query regression (Recall@6 77.45% to 73.53%). Query-description guidance
alone has not made first-search recovery reliable.

### Trial 1 judge comparison

The unchanged v3 judge reused its matching **15/15 passing saved calibration**
from `component-5-reliability-v3-scoped-before.json`, with file-hash provenance.
Trial 1 received one accepted grade, no validation repairs and no material errors.
Trials 2 and 3 were intentionally not judged; their semantic metrics remain unscored.

| Metric on the identical question | Saved accepted grade | Trial 1 accepted grade |
| -------------------------------- | -------------------- | ---------------------- |
| Correctness | 100% (1/1) | 100% (1/1) |
| Completeness | 100% (3/3) | 100% (3/3) |
| Groundedness | 96.30% (26/27) | 100% (23/23) |
| Citation precision | 85.00% (34/40) | **80.77% (21/26)** |
| Citation coverage | 74.07% (20/27) | 86.96% (20/23) |
| Citation validity | 100% (20/20) | 100% (14/14) |

The precision regression remains visible: two literal citation-number examples and
three citations in a limitation paragraph are not assigned supporting claims by
the judge. The commit statement and two factual conclusions lack directly attached
citations. Valid references therefore do not imply complete citation attachment.

Payload reduction, citation validity and trial-1 reference completeness targets
pass; all-three first-search recovery does not. The failure, precision regression,
extra comment reads and increased output/reasoning usage are retained. No baseline
answer substitution, judge recalibration, permissive fixtures, additional examples,
larger budgets, reranker/model changes or further live evaluations were applied.

## Offline parent-window diagnosis (2026-10-16)

Production code remains unchanged in this follow-up. It diagnoses the failed v5
first search and proposes a bounded parent-completion change. There were **zero
generation, judge or embedding-provider calls**. Candidate capture used the
dedicated evaluation role, a verified frozen index, read-only repeatable-read
transactions and existing validated embedding vectors. Missing vectors would fail
instead of contacting a provider. Subsequent comparisons used the captured files
and candidates in memory; no production database or Redis entries were modified.

### Confirmed cause

For v5 trial 3's query, `streaming source cards retrieved candidates citations chat`,
production `_parent_documents` stops after the first six fused candidates create
six parent windows. All six seed chunks remain visible; this is not a case of the
character limit cutting a selected child out of its parent.

| Fused rank | File | Child lines | Individual expanded parent |
| ---------- | ---- | ----------- | -------------------------- |
| 1 | `repository-chat.tsx` | 1-128 | 1-208 |
| 2 | `chat.py` | 209-296 | 129-376 |
| 3 | `repository-chat.tsx` | 120-259 | 64-313 |
| 4 | `chat.py` | 1-104 | 1-184 |
| 5 | `repository-chat.tsx` | 412-481 | 332-512 |
| 6 | `retrieval.py` | 209-314 | 136-385 |
| 13 | `chat.py` | 373-446 | 293-446 |
| 14 | `chat.py` | 289-383 | 209-446 |

Ranks 13 and 14 are inside the existing 18-candidate pool but are never processed.
Their parents cover the backend citation-card loop omitted from the six outputs.
The final annotated overlap is 20%, with Recall@6 0%. Full-pool reordering remains
rejected; no vector/keyword/RRF settings or candidate rankings changed here.

### Fixed offline hypotheses

Two general hypotheses were evaluated against the same captured candidates:

1. Continue processing the 18 candidates after six parents are selected, allowing
   only the existing full-padding merge rule. Keep the six parents and their order.
   This grows some context but **does not recover trial 3**.
2. Also allow later children from selected files to fit into an existing parent
   by adjusting surrounding padding. Retain the selected child spans, six-parent
   limit, order, line limit and character limit. This recovers trial 3 but initially
   removes some previously visible surrounding lines in two saved queries and in
   trial 3, despite retaining every previously visible child chunk in the pool.

The second prototype therefore gained an annotation-independent guard: compare
the combined original/proposed visible ranges and restore original windows until
all original visible context is retained. A restoration can require another
restoration, so the guard repeats for at most six windows. This keeps the useful
backend expansion in trial 3 and restores its unrelated retrieval window.
It does not consult reference facts or evidence annotations. Production would
capture the initial windows in memory from already-loaded files, not make a second
database fetch. The source prototypes are diagnostic artifacts, not a shipped path.

### Cached comparison results

The 17 saved first queries come from
`component-5-reliability-v3-scoped-full.json`, restricted to questions with required
code evidence. The three v5 trial queries are reported separately; repeated
`mixed-01` queries are not combined with the 17-query macro score. Selection receives
only candidates and source files; the unchanged scorer receives annotations after
selection. Every scored line was validated against its actual frozen file text.

| Measurement | Production reference | Guarded prototype |
| ----------- | -------------------- | ----------------- |
| Saved queries scored | 17 | 17 |
| Recall@6, case macro | 77.45% | 77.45% |
| Precision@6, case macro | 38.24% | **36.27%** |
| Queries with Recall@6 regression | 0 | 0 |
| Original visible lines lost | 0 | 0 |
| Total returned characters, saved queries | 648,493 | 659,960 (+1.77%) |
| v5 trial query Recall@6, trials 1/2/3 | 100% / 100% / 0% | 100% / 100% / 100% |

The guarded trial-3 backend window becomes `chat.py:197-446`, accompanied by
`chat.py:1-216`; all originally visible backend lines remain available. The original
`retrieval.py:136-385` window is retained by the guard. Its total context increases
from 42,405 to 45,927 characters (+8.31%). These are deterministic retrieval results
for saved queries, not new generated answers or new execution-completion scores.

The two precision regressions were inspected individually:

- `architecture-02`: 5/6 to 4/6. A retrieval window moves from lines 136-385 to
  167-416, so it no longer individually contains the complete vector-insert unit
  at 150-169. Another retained window at 26-275 still contains that unit. The
  required evidence and all original visible context remain available.
- `mixed-02`: 3/6 to 2/6. A retrieval window moves from 26-275 to 1-250, so it no
  longer individually contains the complete required range 246-258. Two other
  retained windows still contain it. Again, no original visible context is lost.

These are reductions in duplicated complete-unit coverage per window, not lost
requested facts on the scored queries. They remain real Precision@6 regressions
under the unchanged definition; the evaluator was not changed to dismiss them.
Other existing incomplete-recall cases remain incomplete. The prototype does not
claim an overall Recall@6 improvement or resolve missing evidence in new files
outside the six selected parents.

### Verification and implementation recommendation

Seven synthetic checks passed for late-child completion, ordinary padding merge,
refusal to evict evidence for distant children/new files, text budgets, protection
against padding-induced child clipping and unchanged reranking parent behavior.
Five additional guard checks passed for safe expansion, restoration, overlap,
cascading restoration and file isolation. Across all 20 captured queries, the
guarded results keep at most six windows, at most 250 lines/16,000 characters per
window, valid exact source text and every originally visible line. Neither inputs
nor frozen datasets, fixtures, scorers, calibration or prompts were modified.

Artifacts under `.eval-artifacts/`:

- `parent-selection-v6-trial3-candidates.json` and `-trace.json`: initial exact trace.
- `parent-selection-v6-saved-candidates.json`: frozen candidates/files for replay.
- `parent-selection-v6-comparison.json`, `-review.json`, `-guarded.json`: both
  hypotheses, surrounding-context audit and conservative guard results.
- `diagnose_parent_selection_v6.py`, `compare_parent_selection_v6.py`,
  `check_parent_selection_v6.py`, `check_coverage_guard_v6.py`: reproducible scripts.

Recommend the **guarded child-preserving completion** as the next implementation
candidate, with its precision tradeoff explicit. Keep it inside `retrieval.py`:
retain the first six parents and their ranked seeds; process the rest of the
existing candidate pool only for those files; apply bounded merging and the
visible-context guard; preserve native provenance when a window is restored.
Reranking behavior, prompts, models, budgets, router and scorer definitions stay
unchanged. If implemented, version the code-context cache namespace so old cached
parent selection cannot mask the new behavior; retain cache-aside authorization,
snapshot identity and expiry, without flushing Redis.

Add focused production regressions for completion after capacity, selected-child
and full visible-context retention, cascading rollback, limits and stable parent
order/provenance. Then run affected retrieval/code-tool checks and only selected
live cases to establish whether the improved first search actually removes follow-up
reads and model calls. The current offline comparison **cannot establish lower
answer latency, answer quality or generalization**. Context grows slightly, so a
latency improvement would need fewer investigation calls to outweigh that cost.
No full suite, full benchmark, live generation or recalibration ran in this follow-up.

## Guarded parent completion implementation (2026-10-16)

Implemented the approved conservative completion in production retrieval. Focused
code checks pass, but **live quality validation does not pass**: four of five
executions completed, and only one of three selected answers has an accepted grade.
First-search recovery remains two of three new `mixed-01` trials. All failures are
retained; no answer substitution, prompt changes, gate relaxation or tuning loop
followed these observations.

### Change and focused verification

- Freeze the first six parent documents after actual whole-line character clipping.
  Continue through the same existing candidate pool without changing parent order
  or adding parents. Later chunks from selected files may use bounded padding while
  preserving ranked children. Restore whole original documents, including scores
  and source provenance, until all originally visible text remains available.
- Reuse one renderer for the original snapshot and final parents. Both use files
  from the same single database fetch; completion opens no connection or transaction.
  Reranked retrieval retains its original full-pool behavior.
- Advance code-context keys to `repomind:code-context:v4`. Old v2/v3 entries remain
  until ordinary expiry and are not consumed. Cache identity, TTL, authorization,
  snapshot checks, request-scoped labels and Redis failure/cancellation behavior stay
  unchanged. Search and file-read cache namespace isolation are both tested.
- **64/64 targeted tests passed**: `test_retrieval.py`, `test_code_tools.py` and
  `test_context_cache.py`. Eight new retrieval cases cover completion after capacity,
  one-fetch ownership, child/text limits, full visible-context retention, cascading
  rollback and provenance. Four namespace combinations replace the old single test.
  Changed-file Ruff/formatting passed for four Python files; mypy passed for the two
  changed application modules; diff whitespace checks passed.

The changed production method was also replayed over the existing 20 captured query
rankings/files, with **no database, embedding, model or judge calls**. Every result
matches the guarded prototype in exact text and provenance; every invocation uses
one file fetch, and reranked parent output matches the committed pre-change method.
The 17-query macro Recall@6 remains **77.45%**, Precision@6 is **36.27%** versus
38.24% before, and no original visible lines are lost. The three saved v5 trial
queries all recover their required evidence. This controlled result concerns those
saved queries, not future agent-generated queries.

No full suite, opt-in integration suite, full benchmark, build, lock verification or
judge recalibration ran. The selected live executions exercised the isolated real
evaluation database and production graph. Prompts, models, reasoning settings,
budgets, routing strategies, streaming, scorer definitions and frozen data have no
change in this implementation.

### Five selected live executions

All captures share the same application hash. The first three are repeated
`mixed-01` executions; the other two are the precision-affected controls. Counts are
**three complete workflows, one qualified partial and one failed workflow**.

| Capture | Execution / workflow | Recall@6 | Final coverage | References | Seconds | Model/tool calls |
| ------- | -------------------- | -------- | -------------- | ---------- | ------- | ---------------- |
| `mixed-01` trial 1 | Completed / complete | 100% | 100% | 18/18 valid | 35.937 | 6 / 3 |
| `mixed-01` trial 2 | Completed / complete | 100% | 100% | 11/11 valid | 33.563 | 6 / 3 |
| `mixed-01` trial 3 | Completed / complete | **0%** | 100% | 15/15 valid | 44.203 | 7 / 5 |
| `architecture-02` | **Error / failed** | Error | Unscored | **Rejected citation** | 46.360 | 6 / 5 |
| `mixed-02` | Completed / incomplete | 100% | 100% | 17/17 valid | 23.610 | 7 / 5 |

The 61/61 validity count covers the four completed responses only. It does not
include the rejected answer as a valid response. No unmatched fixtures occurred.
`mixed-02` retains its actual PR-diff HTTP 503 and deterministic partial qualification;
its known requested comparison remains answerable from observed PR body/code evidence.

Recorded first queries:

1. `chat source cards retrieval citations answer`
2. `chat source cards retrieved sources citation`
3. `chat source cards citations retrieved sources`

Each mixed trial reads issue details and makes two initial concurrent code searches.
Trials 1 and 2 stop without file or comment reads. Trial 3 additionally searches
`parse citation markers from answer text filter cited sources` and reads
`apps/api/app/chat.py:380-446`. It recovers the requested backend evidence then;
this later coverage is not Recall@6. Full arguments and observed records remain in
the raw reports.

### Remaining first-search miss

One targeted read-only diagnostic replayed trial 3's new failed query using its
already cached vector and verified frozen files. It reproduced the live tool output
exactly. The six selected parents include only the beginning of `chat.py`, seeded
by lines 1-104. Later chunks 373-446 (rank 8) and 289-383 (rank 11) are both inside
the pool and individually expand to sufficient evidence. However, neither can share
a 250-line contiguous parent with the selected seed. Completion cannot add or replace
a parent under this approved policy. Removing the context guard also fails to recover
the evidence, so the guard is not responsible for this miss.

This is a remaining parent-selection/coverage limitation, rather than vector recall
failure, character clipping or a malformed query. It would require a separately
evaluated selection change to solve; no broader ranking or budget change was made.

### Latency and tokens on the three repeated mixed runs

| Measurement | Saved v5 trials, median | New trials, median |
| ----------- | ----------------------- | ------------------ |
| Workflow seconds | 44.562 | 35.937 (-19.36%) |
| First answer delta seconds | 42.500 | 33.516 (-21.14%) |
| Completed model calls, including router | 7 | 6 |
| Tool calls | 4 | 3 |
| Input tokens | 63,190 | 63,779 (+0.93%) |
| Output tokens, including reasoning | 5,182 | 2,743 (-47.07%) |
| Reasoning tokens | 3,675 | 1,117 (-69.61%) |

Per-trial input/output/reasoning counts are 61,263/2,456/758,
63,779/2,743/1,117 and 103,804/4,073/2,493. First deltas arrive at 33.516s,
30.563s and 42.891s respectively. Observed synthesis spans are 7.671s, 9.000s and
13.859s. The first pair of searches overlaps at approximately 9s each, while later
searches are faster. Native timing records retain every stage; overlapping tool
durations must not be added as workflow time.

The new runs have no GitHub comment reads, whereas two saved v5 runs do. Reasoning
output also varies substantially despite unchanged settings. Therefore the observed
latency reduction cannot all be attributed to parent completion. First-search
recovery is still 2/3, and trial 3 still needs a file read. Controls use older saved
baseline application revisions and are reported separately, not as a unified causal
latency comparison: `architecture-02` rises from 39.265s to 46.360s and fails;
`mixed-02` falls from 25.921s to 23.610s and remains a qualified partial.

### Execution and judge failures

`architecture-02` first omits its terminal marker; the existing recovery succeeds.
Its final answer then invents a repository `/tree/<commit>` URL absent from the
observed citation catalog. Strict citation validation rejects it with
`citation_invalid`; the workflow answer is cleared. The raw specialist finding,
timings, recovered `final_marker_missing` and unrecovered citation error remain
available. The saved baseline completed; this new failed execution is an observed
regression, without proof that the retrieval change caused it. The validator was
not loosened to accept a plausible but unobserved URL.

The unchanged v3 judge reused the exact matching saved 15/15 passing calibration.
Of three selected answers, **two were attempted, one accepted, one rejected; one
was skipped because execution had already failed**:

- Trial 1 `mixed-01`: grading fails with `Cited prose must be classified as claims
  or limitations.` The configured three total validation attempts all fail. Semantic
  metrics remain errored; trial 2 or 3 was not substituted. Existing diagnostics do
  not retain the failing section's raw label, so its exact section cannot be identified
  without another judge call. No such call or evaluator change was made.
- `architecture-02`: not judged because generation/citation validation failed.
- `mixed-02`: accepted, no material errors, all three reference facts matched.
  Correctness/completeness remain 100%, groundedness is 96% (24/25), citation
  precision 77.27% (17/22), coverage 60% (15/25), validity 100% (17/17).
  Its saved accepted grade had groundedness 100%, precision 100% and coverage
  88.89%; these regressions remain visible. The new unsupported label concerns a
  provenance/methodology sentence, with the judge itself describing it as an
  interpretive limitation rather than a domain fact. This possible label inconsistency
  is retained, not corrected after seeing the score.

There is no accepted new quality grade for `mixed-01` or `architecture-02`; their
saved grades cannot establish current answer quality. The implementation passes
its focused correctness checks but does **not** establish a new quality acceptance
result, whole-benchmark score or generalization claim.

Artifacts use `.eval-artifacts/parent-completion-v7`: `-offline.json`,
`-trial-{1..3}.json`, `-controls.json`, `-judged.json`, `-pipeline.json`,
`-summary.json`, separate `-mixed-comparison.json`/`-controls-comparison.json`, and
`-trial-3-diagnosis.json`. Comparison baselines preserve their separate application
revision provenance. All failed captures and rejected grades are retained. No
additional live reruns, recalibration or automatic tuning followed the failed gates.

## Latency-first follow-up (2026-10-16)

The saved v7 traces were inspected before changing execution. Warm code searches
took about a second; specialist model turns and synthesis dominated. The failed
CODE control also used an extra 19.078s answer generation after a missing final
marker. Neither nonempty retrieval nor a source-card count proves that an
investigation can safely stop.

Changes in this pass:

- Offline timing records now include nested retriever runs, model phase, message
  character counts, first nonempty callback token time, output kind and provider
  usage linked to each native run ID. Failure/cancellation records remain intact.
  No bodies, tool arguments or credentials are added to timing records. Retriever
  spans include embedding, SQL and assembly; they do not isolate those costs.
- Specialist model input keeps each first observed code window in full and removes
  only later exact ID/content duplicates. References retain metadata, and native
  artifacts, execution records, citations and full findings remain unchanged.
  Conflicting text, failures and native bodies differing from their artifacts
  remain full. Synthesis uses the same exact-match removal rule with its catalog.
- MIXED summaries request concise supported findings rather than a second polished
  answer. Repeated synthesis instructions are consolidated into the shared final
  answer prompt. Completion reminders apply to direct and internal specialists,
  while necessary comments, callers and callees remain available.
- A measured scheduling regression exposed an ambiguous router example that used
  sequential execution despite naming the indexed subject. The dependency rule
  now permits parallel investigation of a named component, symbol or behavior
  without an exact path. The existing sequential example instead requires GitHub
  to reveal unknown code targets. No example, intent or planning call was added.

Models, reasoning settings, budgets, native SSE gating, evidence windows, retrieval,
reranking, repository authorization, session ownership and Redis behavior were not
changed. The final-marker repair remains available; a prompt reminder cannot
guarantee that a provider will follow the protocol.

### Selected executions and retained regressions

Four initial executions were followed by one execution of the affected scheduling
case. All **5/5 completed**, with **0 fixture misses** and **67/67 valid citation
references**. These are execution/citation-validity checks, not semantic grades.
The initial and scheduling captures retain their separate application hashes.

| Capture | Route | First answer token | Workflow | Model/tool calls |
| --- | --- | ---: | ---: | ---: |
| Initial mixed-01 trial 1 | sequential MIXED | 58.922s | 61.094s | 6 / 3 |
| Initial mixed-01 trial 2 | sequential MIXED | 63.797s | 65.359s | 6 / 4 |
| architecture-02 control | CODE | 23.578s | 35.375s | 4 / 5 |
| github-02 required-comments control | GITHUB | 8.266s | 8.782s | 3 / 1 |
| mixed-01 after scheduling fix | parallel MIXED | 51.000s | 52.656s | 7 / 5 |

The saved failed CODE control took 38.860s to its first delta and 46.360s overall,
with six model calls. The new CODE answer completes without marker recovery;
this comparison involves different outcome statuses and one sample per revision.
The GitHub control directly reads necessary comments and streams its native final
generation. Initial MIXED trial 2 reads unrequested comments in the same batch as
details: it adds evidence/context, but did not add a separate model turn here.

The first two MIXED trials use 57,360 median input tokens versus 63,779 across the
three saved v7 trials (-10.06%), but output tokens rise from 2,743 to 4,202 and
reported reasoning tokens from 1,117 to 2,727.5. Their median first delta worsens
from 33.516s to 61.360s. Both initial regressions are retained. These small samples
have different queries and scheduling, so they do not isolate a prompt's effect.

Both fictional routing probes authored during this pass succeed: a named renderer's conversion
behavior uses PARALLEL, while an issue with unknown affected functions uses
GITHUB_THEN_CODE. The follow-up graph genuinely overlaps investigations: CODE runs
from 1.531s to 26.500s, GITHUB from 1.531s to 7.485s, and synthesis starts at 26.500s.
Two missing requested implementation windows require file reads; they are retained.
Synthesis then takes 26.156s, with its first callback text at 24.500s and 3,287
reported reasoning tokens. Callback timing cannot separate provider queuing,
prefill and hidden reasoning time.

**MIXED latency is not resolved.** Its follow-up first delta remains slower than
the saved 33.516s median. Cold retrieval spans are 7.3-11.4s versus roughly 0.8s
warm, and final synthesis remains the largest single stage. All three selected
MIXED executions have Recall@6 = 0 and final evidence coverage = 100%; the CODE
control has 75% and 100%, respectively. First-search misses remain visible and
necessary follow-ups are not blocked to improve latency. No new whole-benchmark
or judged quality score is claimed.

Validation from `apps/api` with the installed environment:

- `pytest -p no:cacheprovider tests/test_agents.py tests/test_orchestration.py tests/test_chat_streaming.py`: **154 passed**.
- `pytest -p no:cacheprovider tests/test_evals.py -k 'execution_timing or retrieval_timing'`: **12 passed**, 62 deselected.
- After the routing-only follow-up, `pytest -p no:cacheprovider tests/test_orchestration.py`: **58 passed**.
- Changed-file Ruff and formatting checks passed. Mypy passed for `app/agents.py`,
  `app/orchestration.py` and `app/evals/execution.py`.
- API build equivalent `python -m compileall -q app` passed. Generated bytecode
  was removed after verifying the three cache paths and their contents.

No full suite, full benchmark, judge, recalibration or saved 17-query retrieval
replay ran. Artifacts: `.eval-artifacts/latency-v8-{trial-1,trial-2,controls}.json`,
`latency-v8-scheduling{,-probes}.json`, `latency-v8-pipeline.json`, and
`latency-v8-summary.json`. The summary includes every execution and both routing
probes. No further provider runs followed the unresolved latency result.

## Matched synthesis and cold-retrieval probes (2026-10-16)

This follow-up tests reasoning effort against identical saved evidence and profiles
retrieval separately. Production defaults and application code are unchanged.
Hive documents `none`, `low`, `high` and `max` for the configured model, so the
proposed medium comparison was corrected to **high versus low** before calling
the provider. See [Hive's model parameters](https://docs.thehive.ai/docs/deepseek-v41-flash).

### Matched synthesis

The saved scheduling report contains all twelve observed code windows, five tool
records and both specialist findings. Its original model-selected source arrays
and complete native prompt were not retained. The diagnostic therefore rebuilds
the catalog from actual results, derives selections from specialist citations and
keeps normalized saved tool arguments. Every code window is verified against the
frozen commit. This is a matched reconstructed input, not an exact replay of the
historical request. No old final answer, reference facts or evidence annotations
are passed to synthesis, and no investigation, router or database call is rerun.

The actual native messages have the same SHA-256 in both arms:
`50ee8465018b2a442492b7831b31e6f4791cb46bba7a7ede7e49a32d3925f7f9`.
They contain 105,459 characters and both provider responses report 31,560 input
tokens with zero prompt-cache-read tokens. The only request parameter changed is
`reasoning_effort`. The model, prompts, 8,192 thinking-token cap, 16,384 completion
cap, 120s deadline and SDK retry setting remain fixed. Order is high then low, one
completed invocation per setting; this is descriptive, not a robust average.

| Synthesis setting | First answer token | Total with validation | Output tokens | Reasoning tokens | Valid references |
| --- | ---: | ---: | ---: | ---: | ---: |
| high | 19.547s | 19.813s | 2,846 | 2,296 | 15/15 |
| low | 13.547s | 15.531s | 2,004 | 1,482 | 14/14 |

Low reduces observed first-answer latency by **30.70%** (6.000s), total synthesis
time by **21.61%**, and reported reasoning tokens by **35.45%**. Visible output
tokens are 550 versus 522. The responses produce 157 and 168 native answer deltas,
respectively; buffered and validated text match, and the production citation
validator plus offline reference validation pass. Citation validity does not prove
that every claim is supported or that the citation set is minimal.

Qualitative coding-agent review finds the same three required facts in both:
the issue reports unused cards, the indexed backend emits retrieval-derived
citations without matching answer markers, and card presence alone establishes
neither citation usage nor claim support. The low answer adds a redundant `[3][9]`
reference to overlapping frontend windows; both include the handler at lines
314-317, and source 9 suffices. No calibrated judge, independent review, semantic
scores or whole-benchmark quality claims are supplied by this check.

Two earlier synthesis invocations failed in the diagnostic harness because the
streaming helper was called without a parent Runnable context. Both failures are
retained in `latency-v9-synthesis.json`, with no emitted answer delta or returned
provider usage. An offline regression reproduces the missing-parent error and
verifies that a RunnableLambda with explicit child config forwards both test
chunks on Python 3.10. Only the two synthesis calls are then repeated in the
corrected harness. Their input hashes also match the failed attempts. Across all
attempts there are **4 synthesis invocations: 2 completed, 2 harness failures**;
failed-stream token usage is unavailable. These failures are not included as
successful latency samples or concealed as production agent outcomes.

### Cold retrieval

Two identical uncached retriever invocations use the dedicated unprivileged eval
connection and read-only transactions. The existing retriever and embedding
implementation run with instrumentation around embedding, HTTP and SQL calls;
no cache, retrieval parameter or model option changes. The local model is naturally
unloaded before the first call and resident before the second. No unload, service
restart or forced prewarming is performed. Database pages are not flushed.

| Stage | Naturally cold model | Resident model |
| --- | ---: | ---: |
| Complete retriever | 5.135s | 0.482s |
| Query embedding, including HTTP setup | 5.015s | 0.425s |
| Ollama-reported model loading | 4.881s | 0.338s |
| All measured SQL | 0.108s | 0.051s |
| Parent assembly, including its SQL | 0.006s | 0.002s |

Connection setup and role validation take 0.059s outside these retriever spans.
HTTP client initialization takes about 0.006s in each call. Nested events must not
be summed with their parent spans. Ollama's durations are converted from
nanoseconds as documented in its [embedding API](https://docs.ollama.com/api/embed).
Pool acquisition, request authorization and Redis behavior are outside this
microprofile; it does not replace production request timing.

Model loading accounts for about 95% of the observed cold retriever time. SQL is
not the dominant cost in this probe. The resident 8,192-context model reports
2.86GB of VRAM use, so a longer residency policy should retain an idle expiry.
The current embedding request supplies no explicit `keep_alive`; the service's
configured expiry applies. Keeping the model resident can reduce subsequent
reloads, but cannot remove the first-ever load while preserving lazy startup.

Both retrieval calls complete and every returned window matches frozen source.
The same six windows/content hashes occur in both, but ranks 3 and 4 swap.
The strict ordered-window equality check therefore fails and the original profile
returns exit code 1. That check remains false in the saved capture; a separate
unordered-set comparison is true. No new retrieval metric or explanation for
the rank variation is inferred from this timing probe.

### Scope and recommendation

Low reasoning for tool-free synthesis is a promising targeted change; keep
investigation reasoning unchanged. Before changing its production default, assess
the saved low answer with the existing quality methodology and additional distinct
saved investigations, without rerunning their agents or searches. A bounded
embedding-model residency policy addresses the measured cold-start cost without
rewriting SQL, changing retrieval or adding a cache layer.

These stage measurements do **not** establish a new full-workflow response time,
whole-benchmark score or held-out generalization. No further model runs are made
after the completed pair. Production reasoning, lazy embeddings, authorization,
Redis, routing, streaming and tool/session lifetimes remain unchanged.

Validation is limited to diagnostic preflights, the offline callback regression,
the two corrected synthesis streams, two read-only retrieval profiles, script Ruff
unused-name checks and formatting. No application tests, build, mypy, full
benchmark, recalibration or judge are rerun because no application source changed.
Artifacts are `latency-v9-synthesis.json` (failed harness),
`latency-v9-synthesis-runnable.json` (corrected pair), `latency-v9-retrieval.json`,
and `latency-v9-summary.json`, with the three diagnostic scripts alongside them.

## Bounded latency rollout v10

### Implementation and validation scope

Only the two authorized production optimizations are applied:

- `SYNTHESIS_REASONING_EFFORT` defaults to `low` for Hive MIXED synthesis in
  production and full graph evaluation. `high` restores the previous effort.
  Supported values follow the provider's `none`, `low`, `high`, `max` contract.
  Specialists retain `high`, the router retains `none`, and explicit OpenAI
  provider behavior, token limits, thinking cap, deadlines and retries are unchanged.
- Every Ollama embedding request supplies a positive finite idle residency,
  `OLLAMA_EMBEDDING_KEEP_ALIVE_SECONDS=300` by default, encoded as `keep_alive="300s"`.
  The model, pinned digest, query instruction, dimensions, context limit,
  truncation policy and vector-space fingerprint are unchanged. No reindexing or
  cache namespace change is required. This policy may match Ollama's existing
  default; it does not remove the first cold load or prove a separate latency gain.

The settings are recorded in evaluation metadata. Native streaming, citation
validation, routing, tool selection, request authorization and Redis behavior
are unchanged. No prompt, retrieval, reranker, model or agent-budget tuning is
performed. Existing uncommitted work from earlier passes is preserved.

The [Hive model documentation](https://docs.thehive.ai/docs/deepseek-v41-flash)
and [Ollama embedding API](https://docs.ollama.com/api/embed) were rechecked before
implementation. Initial validation runs 123 affected deterministic tests, all
passing. Mypy passes all 25 API source files; the API compileall build and
`uv lock --check --offline` pass. Ruff finds one new overlong metadata line,
which is corrected and checked again successfully. Changed-file formatting
passes. A full formatting check also reports four pre-existing untouched files:
`test_config.py`, `test_github_mcp_server.py`, `test_repositories.py`, and
`test_worker_github.py`; these are left unchanged.

### Single MIXED rollout smoke

One `mixed-01` workflow and its existing calibrated v3 judge are run. Calibration
is reused only after exact judge-configuration validation against the saved
15/15 passing calibration. Neither calibration nor answer generation is repeated.

| Measurement | Observed |
| --- | ---: |
| First actual user-facing answer delta | 39.094s |
| Complete workflow | 41.594s |
| Synthesis call | 7.844s |
| Synthesis first native content token | 5.344s |
| Model calls including router | 8 |
| Tool executions | 5 |
| Workflow input / output tokens | 96,214 / 3,040 |
| Workflow reasoning tokens | 1,391 |
| Synthesis input / output / reasoning tokens | 27,784 / 843 / 322 |
| Correctness / fact completeness / groundedness | 100% / 100% / 100% |
| Citation validity | 13/13 (100%) |
| Citation precision / coverage | 88.89% / 66.67% |
| First-search Recall@6 / Precision@6 | 100% / 16.67% |
| Final evidence coverage | 100% |

The investigation makes two code searches, two file reads and one issue-body
read. It returns complete findings without tool errors, missing requested facts,
judge validation errors or material answer errors. Both specialist evidence and
live/indexed provenance remain present. Citation quality is imperfect: some
supported claims lack attached citations and one attaches prompt evidence to a
persistence claim. Against the saved same-case v3 grade, citation precision rises
from 85.00% and coverage from 60.87%, while correctness, completeness,
groundedness and validity remain 100%. No material smoke regression is observed;
the new low synthesis default is retained.

The previous v8 parallel workflow took 52.656s with first answer at 51.000s;
this smoke takes 41.594s with first answer at 39.094s. These are different tool
trajectories and contexts, not a controlled reasoning-only comparison. Retrieval
spans still take about 8.3s and CODE investigation dominates the remaining
pre-answer latency. The earlier v9 identical-input comparison remains the
controlled evidence for the synthesis setting, rather than this one smoke.

Artifacts: `latency-v10-smoke.json` preserves execution independently of grading;
`latency-v10-smoke-judged.json` preserves the accepted quality grade. The user's
final instruction additionally authorizes one full API suite and one unchanged
24-case graph evaluation after the successful smoke, without a tuning loop.
The full API suite, with `REPOMIND_EVAL_INTEGRATION=1`, passes **456/456** tests
in 111.03s with no skips or failures. Full benchmark results follow separately.

### Requested full benchmark and unchanged judge

Exactly one unchanged 24-case execution run follows the smoke. It returns
**23/24 benchmark execution completions (95.83%)**, followed by **20/23 accepted
eligible judge grades**, or **20/24 attempted cases (83.33%)** with accepted grades.
There is one execution failure and three grading failures. The process exits 1;
the 24/24 execution and grading acceptance gates are **not met**. Neither failed
answers nor rejected grades are substituted or retried through another run.

| Category | Attempts | Execution completions | Accepted judge grades |
| --- | ---: | ---: | ---: |
| Single file | 6 | 6 | 3 |
| Architecture | 6 | 6 | 6 |
| Insufficient evidence | 4 | 4 | 4 |
| GitHub | 4 | 4 | 4 |
| MIXED | 4 | 3 | 3 |
| Total | 24 | 23 | 20 |

`mixed-03` reads issue #103 successfully, then unnecessarily requests its comments.
That method has no matching strict fixture. The GitHub investigation retains
`Unmatched deterministic fixture call.`, and the synthesizer produces a qualified
partial answer using actual issue-body and indexed-code evidence. The evaluator
correctly rejects the unmatched request. The failing selection occurs in the
unchanged GitHub specialist before synthesis; the fixture is not broadened.

`code-02` and `code-03` exhaust the existing judge validation attempts because
cited prose is labeled nonfactual. `code-05` exhausts those attempts because a
quote is not found within its assigned section. All three are CODE-only cases
using unchanged specialist reasoning; these are evaluator validation failures,
not accepted incorrect-answer grades or synthesis citation failures. No new
judge retry policy is added. `architecture-06` and `missing-02` recover from
validation errors through the judge's existing retry behavior; their errors
remain in the artifact. The execution-failed case is not eligible for semantic
grading under the existing runner contract.

The workflow reports 18 complete and six incomplete investigations, including
the fixture-failed one. Qualified partial completions are not relabeled complete
investigations. Observed controlled errors include an invalid code-tool range,
missing terminal declaration, unavailable GitHub resource, deferred broad
searches, invalid investigation JSON, a tool budget limit, expected fixture 503s
and blocked unchanged requests. Full native execution records retain them.

| Metric | Current macro score | Scored cases |
| --- | ---: | ---: |
| Recall@6 | 81.86% | 17 |
| Precision@6 | 40.20% | 17 |
| Final evidence coverage | 100% | 17 |
| Answer correctness | 100% | 18 (two additional cases not applicable) |
| Fact completeness | 100% | 18 (two additional cases not applicable) |
| Groundedness | 97.44% | 20 |
| Citation precision | 92.03% | 19 (one additional case not applicable) |
| Citation coverage | 73.92% | 20 |
| Citation validity | 100%, 353/353 references | 23 (one case not applicable) |
| Correct abstention | 100% | 4 |
| Unnecessary abstention | 0% | 16 |

Semantic averages cover accepted grades only. They do not establish 100%
correctness or completeness over all 24 attempts. Retrieval and citation-validity
checks also retain the execution-failed case's observed evidence; its semantic
metrics remain unscored. All three eligible MIXED-category answers have accepted
grades, matching every requested reference fact with no material answer error.

Across **all 24 attempts**, median first actual answer delta is **20.406s** and
median workflow duration **25.336s**. Workflow durations by category have medians
of 23.055s (single file), 30.195s (architecture), 25.118s (insufficient evidence),
12.312s (GitHub) and 44.266s (MIXED). The slowest is `mixed-04` at 80.406s with
first answer at 74.422s. Its CODE investigation lasts about 44.6s after fan-out;
it includes four searches, three attempted file reads and a budget error. Its
GitHub branch repeats an unavailable diff request, which the existing guard
blocks. Synthesis then takes 33.609s on 151,435 input characters. These are
observed follow-up targets; no required read is removed to force a latency goal.

All completed native model calls across the 24 attempts consume **1,068,886 input
tokens and 44,572 output tokens**, including **19,641 reasoning tokens** and
**185,088 cached input tokens**. These totals include the failed benchmark attempt
and exclude offline judge calls. Judge token usage is not exposed by the existing
judge instrumentation. Production workflow timings exclude offline grading.

### Historical comparisons and retained regressions

The saved `component-5-reliability-v3-scoped-after.json` has the identical judge
configuration and unchanged benchmark/scorer contracts, but a different
application hash. It predates the intervening context, investigation and retrieval
work. These comparisons are descriptive, not a controlled measurement of the
two v10 settings.

| Metric on identical scored cases | Cases | Saved v3 | Current v10 |
| --- | ---: | ---: | ---: |
| Recall@6 | 17 | 77.45% | 81.86% |
| Precision@6 | 17 | 38.24% | 40.20% |
| Final evidence coverage | 17 | 100% | 100% |
| Correctness | 18 | 94.44% | 100% |
| Completeness | 18 | 97.22% | 100% |
| Groundedness | 20 | 97.50% | 97.44% |
| Citation precision | 19 | 88.53% | 92.03% |
| Citation coverage | 20 | 75.81% | 73.92% |
| Citation validity | 23 | 100% | 100% |

Whole-run comparisons must retain the lost coverage: saved v3 has 24/24 execution
completions and accepted grades, while v10 has 23/24 and 20/24 respectively.
Saved v3's whole-run correctness/completeness are 95.45%/97.73% over 22 applicable
cases; v10's 100%/100% cover only 18. Likewise, saved v3's 97.80% groundedness and
74.20% citation coverage cover all 24, versus v10's 97.44%/73.92% over 20.

The saved whole-run median workflow duration is 22.023s, versus 25.336s now
(+3.313s); first-answer median rises from 16.039s to 20.406s (+4.367s). This run
therefore does not establish a whole-benchmark latency improvement. The earlier
identical-input synthesis comparison and successful rollout smoke support
retaining low synthesis effort; no material smoke regression is observed.
Five-minute embedding residency remains explicit and bounded, with no claim of
an independently measured gain over the service's previous default.

No further trials, prompt tuning, retrieval changes, model changes, fixture
changes, recalibration or full evaluations are started. The next focused work
should use saved traces to address unnecessary comment/diff requests and saved
answers to diagnose judge label/quote validation failures. Preserve strict
evidence checks and regrade only affected saved answers after a separate fix.
Tail latency and citation coverage remain follow-ups, not completed optimizations.
An independent holdout is still required for generalization claims.

Artifacts are `latency-v10-full.json` (execution),
`latency-v10-full-judged.json` (all grading outcomes),
`latency-v10-summary.json` (counts, timing, tokens and paired comparisons), and
the one-shot `run_latency_v10.py` driver and offline `summarize_latency_v10.py`.
Historical reports, benchmark data, strict fixtures and scorer/judge code are
unchanged. Both authorized settings are retained; no legacy fallback is added.

## Targeted tool selection and evaluator repair v11

### Diagnosis and bounded changes

This pass uses the v10 saved executions and rejected grades. It does not run a
new full benchmark, regenerate user-facing answers, change retrieval, increase
agent budgets or loosen fixtures. The old rejected native judge labels were not
retained, so their exact offending sections cannot be proved retrospectively.
However, the saved inputs reproduce concrete fragmentation: `code-02` has a
standalone `documents [9].`, `code-03` has `lists [1].`, and `code-05` splits an
inline query literal across sections. These are produced by the fixed 600-character
word splitter. The old retry feedback also lacks a failing section ID.

Evaluator v4 prefers sentence and line boundaries outside backtick spans, keeping
numeric citations after sentence punctuation attached. A sentence can exceed the
600-character soft target up to the existing 2,400-character batch cap; extremely
long unbroken input still has a bounded fallback. Every nonwhitespace character
and original offset is retained. No answer text, citation reference, evidence
window or validation rule is repaired or dropped. Exact quotations, evidence
IDs, source support, section coverage and rejection of factual cited prose
labeled nonfactual remain enforced. Cited introductions asserting observed source
provenance are clarified as claims in the judge instructions. Validation repair
feedback now names the failing section. Model settings, retry counts and scorers
are unchanged. The evaluator configuration is versioned; v3 calibration cannot
be reused for this implementation.

The GitHub prompt and tool descriptions distinguish a requested report or stated
PR claim, already established by its body, from requested discussion or actual
patch behavior requiring more evidence. The existing fictional patch example is
made conditional on body text that does not establish the actual change. No new
examples, intents, selection classes or benchmark-specific facts are introduced.
After a failed GitHub read, the existing model-call checkpoint explicitly treats
that unchanged request as unavailable for the investigation, including transient
server errors. Invalid arguments can still be corrected, and other requested
resources can still be read. Native errors, artifacts and tool records remain
unchanged. Necessary comment and diff tools stay available, and the existing
execution guard still blocks repeats and security failures.

### Targeted validation

All **205 affected deterministic tests pass**: 174 agent/graph/streaming/grounding
tests, 16 selected judge/calibration tests, then 15 model-factory tests after the
synthesis default restoration below. Regressions cover logical boundaries,
long clauses, inline code, grouped citation references, exact offsets, bounded
unbroken input, strict rejected labels, section-specific errors and failed-read
checkpoints preserving tools and native errors. Changed-file Ruff, formatting,
mypy and compilation pass. One new 89-character configuration line is corrected
after lint reports it; the affected check then passes. No full API suite, frontend
build, lockfile resolution or full benchmark is repeated.

One fresh v4 calibration passes **15/15 existing synthetic calibration cases**;
calibration files are unchanged. Only the three saved failures are regraded:

| Saved case | v3 outcome | v4 outcome | Correctness / completeness | Groundedness |
| --- | --- | --- | ---: | ---: |
| code-02 | Cited prose labeled nonfactual | Accepted, no validation retries | 100% / 100% | 100% |
| code-03 | Cited prose labeled nonfactual | Accepted, no validation retries | 100% / 100% | 100% |
| code-05 | Quote outside assigned section | Accepted, no validation retries | 100% / 100% | 92.59% |

All cited references remain valid. Accepted grading does not imply every extra
claim is grounded: `code-05` has two unsupported extracted claims among 27.
The new diagnostic wrapper retains returned labels and exact input sections for
audit, without changing grading behavior or logging credentials.

Three GitHub-only live-model checks use unchanged strict fixtures and real
per-call repository authorization in the dedicated evaluation database. No
router, Code Agent, retriever, embedding or synthesizer runs. Request scopes and
temporary evaluation memberships are owned and cleaned up by the diagnostic.

| Assignment from saved request | Old model calls / reads | New model calls / reads |
| --- | --- | --- |
| mixed-03 GitHub portion | 3; issue details, unnecessary comments (fixture miss) | 2; issue details only |
| mixed-04 GitHub portion | 4; PR details/files/diff plus blocked repeated diff | 2; PR details only |
| github-02 required comments | 2; comments | 2; comments |

All three return complete findings with no tool errors or fixture misses. The
requested issue report, PR claim and reviewer comment remain present and cited.
Callback-span durations are 10.688s, 12.016s and 7.516s respectively; they are not
full-chat latencies. Total provider usage for these checks is 18,784 input and
1,148 output tokens. Reduced branch work does not establish an end-to-end MIXED
latency gain: the parallel Code Agent may still dominate the critical path.
These are single targeted observations, not repeated reliability trials or a
held-out generalization result.

### Is low synthesis reasoning worth retaining?

No synthesis is regenerated. The two successful v9 answers on identical native
input (31,560 tokens, zero reported cached input for either) are graded by the
same freshly calibrated v4 judge, without exposing their effort labels in judge
inputs. Saved citation references are reconstructed and checked against the
original pair before any model call. Both grades are accepted without validation
errors or material answer errors, and both convey all three reference facts.

| Matched saved synthesis measurement | High | Low |
| --- | ---: | ---: |
| First user-facing delta | 19.547s | 13.547s |
| Complete synthesis | 19.813s | 15.531s |
| Output tokens, including reasoning | 2,846 | 2,004 |
| Reasoning tokens | 2,296 | 1,482 |
| Correctness / completeness / groundedness | 100% / 100% / 100% | 100% / 100% / 100% |
| Citation validity | 15/15 | 14/14 |
| Citation precision | 92.59% | 100% |
| Citation coverage | 91.30% (21/23 claims) | 76.00% (19/25 claims) |

Low reduces this pair's synthesis time by 21.61%, first-answer latency by 30.70%,
output tokens by 29.59% and reasoning tokens by 35.45%. It is a useful latency
and output-token trade-off, but not a clear quality improvement. The low answer
adds uncited provenance assertions and has uncited factual summaries/conclusions;
all are supported by observed evidence, which explains 100% groundedness alongside
lower citation coverage. The high answer also has uncited statements. The judge
extracts 23 versus 25 claims and handles some framing sentences differently, so
the 15.30-point coverage difference is not proof of a general causal effect from
reasoning effort. There is only one matched pair, high was generated first, and
provider variation remains possible.

Given the user's standing requirement to restore high after a material citation
regression, **high is conservatively restored as the synthesis default** in
Settings, environment fallback and the template. Low remains explicitly
configurable through `SYNTHESIS_REASONING_EFFORT=low`; no specialist/router setting
changes. Five-minute bounded embedding residency remains enabled. No further
trials or synthesis prompt tuning is started to chase the coverage score.

The historical full v10 result remains **23/24 executions and 20/23 eligible v3
grades**. The three new v4 grades must not be mixed with v3 grades to claim a new
whole-run score, nor do role-only checks turn the old `mixed-03` failure into a
new full-graph completion. Uniform quality evaluation is deferred to the release
checkpoint, with historical answers graded under the same configuration then.

Artifacts: `targeted-v11-github.json`, `targeted-v11-calibration.json`,
`targeted-v11-saved-failures.json`, `targeted-v11-synthesis-pair.json` and
`targeted-v11-judge-traces.json`; the two one-shot diagnostic drivers are alongside
them. Historical captures, datasets, fixtures, original calibration files and
scorers remain unchanged. The production judge remains absent.
