# Component 5: production graph, streaming, and context reuse

The parent LangGraph workflow is the only production chat and full-evaluation
execution path. The previous single-agent generator, MCP compatibility loader,
and legacy full runner have been removed. The hybrid retriever, frozen corpus,
datasets, scoring definitions, and historical reports remain intact.

## Final-answer streaming decision

CODE and GITHUB stream their specialist's normal final generation. Normal requests
make no additional answer call. The [reliability pass](EXECUTION_RELIABILITY.md)
adds a bounded correction for an unexposed missing-marker failure. MIXED streams the existing Answer
Synthesizer; its specialist summaries remain internal.

Early text chunks cannot prove that a generation is final: the provider may emit
text before a later tool-call chunk. Each final generation therefore begins with
the internal `<final_answer>` declaration. The gate handles a declaration split
across native chunks, removes it, and immediately forwards subsequent text.
Reasoning, unmarked planning, tool arguments, and internal JSON are excluded.
Tool calls after this declaration fail before execution. Missing declarations,
truncated completions, and missing completion signals fail the turn.

| Approach                               | Correctness and complexity                                                                                                                         | Latency and tokens                                                                     |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Direct specialist generation, selected | Explicit terminal protocol plus native callback gate; structured observations come from executed tools. Citation validation remains deterministic. | Reuses the existing final generation and avoids repeating evidence in another prompt.  |
| Dedicated answer generation            | A separate tool-free call gives a simpler role boundary, but still needs completion and citation validation.                                       | Adds one provider round trip and another evidence prompt for each CODE/GITHUB request. |

The selected approach has been exercised through the installed ChatOpenAI SDK,
create_agent, parent graph, and production SSE. Tests pause the provider while
asserting that answer deltas have already arrived. No completed answer is split
into simulated tokens. Python 3.10 requires explicit callback propagation through
nested async calls; the request owns a copied model's callbacks rather than
mutating a shared model.

## Persistence and failure behavior

The server buffers exactly the emitted answer text. After generation it validates
citations against observed evidence and atomically writes the answer, selected
citations, and successful message status. Only then does it emit `citations` and
`done`. Live tool names and heartbeats continue during investigation.

Generation, authorization, citation, and persistence failures emit `error` and
mark the message FAILED. Already emitted text is retained as unverified partial
text without finalized citations. A disconnected browser or upstream send failure
closes the stream, cancels and awaits the owning graph task, and closes scoped MCP
sessions in that task. Failed-message cleanup has a bounded, shielded write; an
unavailable database is logged rather than falsely reported as persisted.

## Redis cache-aside boundary

Only successful indexed-code search results and file windows are shared across
requests. Router decisions, answers, GitHub observations, sessions, and permission
decisions are not cached. A request still authorizes every tool invocation and
verifies its indexed snapshot before reading Redis.

Cache identity includes the backend database namespace, repository, indexed SHA,
`lastIndexedAt`, exact tool arguments, embedding profile, and retrieval/reranking
configuration. Reindexing invalidates keys even when the commit SHA is unchanged.
The TTL defaults to 900 seconds; zero disables caching. Entries are bounded to
256 KiB and each Redis operation to 250 ms, with no registry/cache retries. An
outage, timeout, corrupt value, or oversized entry falls back to the authorized
data layer. Cancellation propagates. Cache hits do not refresh TTLs.

Cached windows contain no request source numbers or tool-call identities. Each
request rebuilds its stable citation catalog from exact paths, ranges, contents,
and commit identity. Embeddings are constructed only for an uncached search.
GitHub-only requests, clarifications, and cached reads do not construct them.

The API lifespan owns the database pool and existing Redis client. Code tools own
connection acquisition and read-only repeatable-read transactions; there is no
registry-wide database lock. Concurrent investigations use independent tool scopes
and connections. No connection stays checked out during final answer generation.

## Evaluation and completion criteria

`python -m app.evals full` evaluates the graph. The independent `retrieval` command
preserves the original retrieval benchmark. Recall@6 and Precision@6 use the first
search batch of at most six windows; final evidence coverage separately uses all
observed searches and file reads. Production imports no offline scorer or judge.
Graph artifacts also record native first-answer-delta timing, workflow duration,
model calls, and provider-reported token usage.

Before removal, the legacy implementation was captured in
`.eval-artifacts/component-5-legacy-final-authorized.json` and its streaming sidecar.
It completed 22 of 24 cases; both insufficient-evidence cases failed and remain in
the report. The accepted historical 24-case baseline is unchanged. Its embedding
and generation settings differ from the current run, so it is not a controlled
latency or quality comparison. Legacy first-delta timing may also include internal
agent text, whereas graph timing records only final-answer text.

Completion requires native streaming tests for all graph routes; hidden internal
output; terminal/completion/citation failures; atomic SQL persistence; disconnect
cleanup; real Redis reuse and revocation; graph concurrency; unchanged benchmark
contracts; and the project's API, web, formatting, type, build, and lock checks.

## Executed validation

- Full API suite with opt-in PostgreSQL/Redis integration: 329 passed.
- After the URL-port regression fix, graph tests: 38 passed, including the new case.
- Ruff, changed-file formatting, mypy (25 source files), web lint/type checks,
  repository build, API lock verification, and diff whitespace checks passed.
- Native SDK tests confirm one router call, one tool-request generation, and one
  specialist final generation in the direct route fixture. MIXED retains its
  existing synthesis call. These are call counts, not measured quality gains.
- The live 24-case graph run completed 15 cases and retained nine errors in
  `.eval-artifacts/component-5-graph-final.json`. Failures include deterministic
  fixture mismatches, exhausted budgets, final-output validation failures, and the
  inline URL-port false positive subsequently fixed and regression-tested. The
  original report remains unchanged. No semantic judge was run for this capture;
  it is not a complete quality baseline or evidence of improvement.
- Router scopes matched the benchmark's evidence-scope annotations in all 24
  cases. Those annotations were inspected offline and were never router inputs.
- The existing comparison command wrote
  `.eval-artifacts/component-5-paired-comparison.json`. Across the 12 cases with
  scored retrieval metrics in both captures, Recall@6 changed by -2.78 percentage
  points and Precision@6 by +4.17 points. These partial paired results exclude
  failed/unscored cases and do not represent whole-benchmark quality.

That initial capture did not establish replacement quality. The subsequent
[execution reliability pass](EXECUTION_RELIABILITY.md) completed 24/24 cases and
records calibrated semantic scores for 23/24 answers, with the remaining judge
failure retained explicitly.

The later few-shot prompt experiment completed 23/24 cases, retaining one strict
fixture failure, and its calibrated judge graded all 23 completed answers. The
same [reliability report](EXECUTION_RELIABILITY.md) records the mixed quality
results, citation-precision regression, judge extraction defect, and development
set limitations. It does not establish a complete quality baseline.

The current router retains only CODE, GITHUB and MIXED, plus clarification and
MIXED scheduling. Tool descriptions guide method selection; no extra GitHub intent
classification or intent-based suppression remains. Router schema validation is
deterministic, while evidence-scope accuracy is evaluated offline. The router's
prompt is not an authorization boundary or a guarantee against prompt injection.
