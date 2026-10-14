# RepoMind quality evaluations

This package benchmarks the production hybrid retriever and shared single-agent
generation function. Cases start with empty conversation history. Results separate
retrieval, groundedness, correctness, citation support, and abstention; there is no
composite score. The fixed GitHub scenarios are synthetic, not real issue/PR history.

See [the executed baseline](BASELINE.md) for the accepted 24-case results,
12-example judge calibration, checks and retained failure cases.
See [retrieval experiments and judge trials](EXPERIMENTS.md) for subsequent
ablations, the chunk-range repair, and cheaper-judge findings.
The latest selected retrieval configuration uses tighter identifier boosts and
24 candidates per branch: Recall@6 **80.88%**, Precision@6 **38.24%** on the fixed
benchmark. These are retrieval-only measurements; earlier full-run grades remain
historical artifacts.

## Setup

Use the existing PostgreSQL/pgvector service and a separate database and login.
For local Docker development, start PostgreSQL and open its administrator console:

```powershell
docker compose up -d postgres
docker compose exec postgres psql -U repomind -d postgres
```

Provision these once; `\password` prompts without storing a password in SQL history:

```sql
CREATE ROLE repomind_eval LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
\password repomind_eval
CREATE DATABASE repomind_eval OWNER repomind_eval TEMPLATE template0;
\connect repomind_eval
CREATE EXTENSION IF NOT EXISTS vector;
```

Set `EVAL_DATABASE_URL` in the ignored `apps/api/.env`, using this dedicated login.
Keep the application connection settings pointed at the application database. The
evaluation login must have no ownership or write grants on application tables.
The runner rejects other database/login names, the application login/database,
superusers, database/role creators, and role memberships. Every corpus write uses
the verified evaluation connection.

From the repository root:

```powershell
pnpm --filter api eval migrate
pnpm --filter api eval prepare
```

`migrate` verifies the target before applying the existing Prisma migrations. It
overrides `DATABASE_URL` only in the migration subprocess. `prepare` reads tracked
Git objects at the dataset's pinned commit, reuses production filtering, chunking,
embedding and index promotion, and preserves an already verified index. It does
not run the background job that opens the application connection.

Preparation needs `OPENAI_API_KEY` for uncached embeddings. Full generation also
needs `OPENAI_CHAT_MODEL`. Cached vectors are validated for dimensions and finite
values. Corpus identities include source hashes, embedding model/dimensions and
actual chunk metadata; changed settings create a separate evaluation index.

## Commands

```powershell
# No generation or judge calls; uncached query embeddings can incur provider calls.
pnpm --filter api eval retrieval --experiment baseline-retrieval

# Isolate either search branch, keeping the same production parent expansion.
pnpm --filter api eval retrieval --retrieval-mode vector --experiment vector-only
pnpm --filter api eval retrieval --retrieval-mode keyword --experiment keyword-only

# Compare 12 versus 24 candidates per active branch, with six final contexts.
pnpm --filter api eval retrieval --candidate-budget 12 --experiment hybrid-12
pnpm --filter api eval retrieval --candidate-budget 24 --experiment hybrid-24

# Check a judge against the known examples before spending on a full run.
pnpm --filter api eval calibrate --judge-model gpt-5.4-mini-2026-03-17 --reasoning-effort medium --timeout 120

# Capture production-equivalent answers and grade with explicit opt-in.
pnpm --filter api eval full --experiment baseline-full --judge-model gpt-5.4-mini-2026-03-17 --judge-reasoning-effort medium --judge-timeout 120

# Capture answers without semantic grading.
pnpm --filter api eval full --experiment capture

# Grade saved answers without regenerating them.
pnpm --filter api eval grade ../../.eval-artifacts/capture.json --judge-model gpt-5.4-mini-2026-03-17 --judge-reasoning-effort medium --judge-timeout 120 --output ../../.eval-artifacts/capture-graded.json

# Compare paired cases with matching gold, corpus, fixtures, scorer and judge settings.
pnpm --filter api eval compare ../../.eval-artifacts/baseline-full.json ../../.eval-artifacts/candidate-full.json --output ../../.eval-artifacts/comparison.json

# Compare different judges on identical saved answers/evidence, with manual review.
pnpm --filter api eval judge-agreement ../../.eval-artifacts/single-agent-v1-accepted.json ../../.eval-artifacts/single-agent-v1-luna-trial.json --output ../../.eval-artifacts/luna-judge-agreement.json
```

Default reports and embedding caches live in the repository's ignored
`.eval-artifacts/`. Explicit relative paths resolve from `apps/api`, where pnpm runs
the package script. Use repeated `--case case-id` flags for a smoke run.
Retrieval mode defaults to `hybrid`: 24 vector plus 24 keyword candidates.
`vector` disables keyword search; `keyword` disables vector search and query
embedding calls. `--candidate-budget 12` or `24` overrides each active branch for
retrieval experiments; omitted budgets use production defaults. The earlier
vector/keyword/hybrid comparison used 12 per branch. All modes
reuse the same fusion and parent expansion with up to six final contexts. The
mode and actual candidate budgets are recorded in reports; full runs remain hybrid.
`--timeout` bounds each retrieval/generation case (default 120 seconds);
`--judge-timeout` bounds each judge request (default 60 seconds). The judge has one
provider retry. Grounding schema/quotation validation can request at most two
corrections, using validation feedback without gold facts; rejections remain in
the report. Ordinary tests never opt into generation, grading or database work.
For reasoning models, explicitly set `--judge-reasoning-effort` (`--reasoning-effort`
for calibration). Sampling temperature is omitted with reasoning enabled, following
[OpenAI's parameter compatibility rules](https://developers.openai.com/api/docs/guides/latest-model).
The selected effort and temperature are recorded in judge configuration.

Generation and tool failures are recorded per case; subsequent cases continue.
Grading runs at most three cases concurrently and checkpoints each completed case.
Calibration also records failed examples and continues. Semantic metrics remain
unscored unless every calibration example passes. A complete quality baseline
requires all dataset cases to finish and receive valid judgments; low scores do
not prevent baseline completion. Reports checkpoint after each completed case.

## Dataset and fixtures

`data/dataset.json` contains version 1: six single-file, six architecture, four
insufficient-evidence, four GitHub, and four mixed questions. Its corpus is RepoMind
at `678cd4ef010ef31f9293aa8ff0c92dafd255d002`. The annotations were source-reviewed
during implementation; the review notes do not assert independent human sign-off.

Each required evidence unit has one or more alternatives; an alternative contains
all source spans needed for that unit. Spans contain exact excerpts and full-file
SHA-256 hashes. Reference facts identify supporting code units or fixture calls.
Additional relevance spans can identify useful evidence beyond the required units.
Mixed reference facts explicitly connect code behavior and GitHub findings.

The loader verifies every source hash, excerpt, reference identifier and fixture
scenario against the frozen corpus. Working-tree changes do not change the corpus.
If a shallow checkout lacks the commit, fetch that Git object before preparation.
For another repository, supply `--dataset`, `--fixtures` and `--corpus-root` pointing
at its local Git checkout. Review annotations and increment dataset/fixture versions
when changing ground truth; preserve the previous reports.

`data/github.json` declares tool names, query/argument variants, complete synthetic
result arrays and controlled errors. Fixture tools retain production MCP schemas,
descriptions, Pydantic argument validation and pagination bounds. Pagination is
applied to fixture arrays. Sorting is irrelevant to the initial empty/single-item
search results; multi-item search fixtures require explicitly declared sort/order
variants with correctly ordered result snapshots. Search query echoes use the
production scope-query builder. Invalid arguments are ordinary recorded tool errors;
unknown resources or undeclared queries are fixture misses, with no live fallback.
A fixture miss makes a run incomplete so missing harness coverage cannot masquerade
as measured answer quality.

## Scoring rules

| Metric | Definition |
|---|---|
| Recall@6 | Fraction of required evidence units fully covered by the union of the final six visible windows. Every span in one annotated alternative must be recovered. |
| Precision@6 | Relevant, nonduplicate windows divided by actual returned windows, up to six. A relevant window completely contains an annotated required or additional relevance unit. Multi-file facts have separately annotated local relevance units where appropriate. |
| Groundedness | Supported domain factual claims divided by extracted domain factual claims, using only evidence actually consumed and successful tool results. |
| Answer correctness | All required reference facts matched with no material factual errors. Fact completeness is reported separately as matched facts divided by reference facts. |
| Citation precision | Supporting valid claim–reference associations divided by all associations plus orphan references. |
| Citation coverage | Domain factual claims fully established by their attached valid citations jointly, divided by all domain factual claims. Joint sufficiency is a validated semantic label; alternative supporting sources need not match the minimal grounding evidence set. |
| Citation validity | Referenced citations resolving to visible source indices/locations, commit-pinned code URLs or structured URLs actually returned by successful tools, divided by referenced citations. |
| Correct abstention | Appropriate acknowledgments of annotated missing information, with all answerable reference facts still addressed and no material errors. |
| Unnecessary abstention | Abstention on an answerable case with an empty expected-missing list. |

Partial span coverage is diagnostic only: it earns no primary recall credit.
Duplicates cannot improve recall and consume precision slots without repeated
credit. Source text is verified against the frozen file; declared but truncated
lines do not count. Retrieval scores are zero for no contexts when code evidence
is required, and N/A when the case has no required code evidence.

Citation parsing recognizes numeric groups such as `[1, 2]`, explicit URLs, and
file locations such as `app.py:10-15`. References inside fenced code are excluded;
numeric literals inside inline code are excluded. Bare file mentions and unused
citation cards are not citations. Code locations must be within evidence actually
seen, not merely somewhere in the repository. Tool-body text cannot manufacture
valid URLs: only structured URL fields from successful calls qualify.
Citation support is restricted to the cited line range or URL-bearing GitHub
entity; unrelated lines or search results cannot rescue a misleading citation.

Every extracted domain factual claim requires evidence. The judge cannot exempt
simple code facts from citation coverage. Pure limitations, absence-of-evidence
acknowledgments and suggestions are excluded from domain claims and evaluated by
the separate abstention task. With no citations, coverage is zero for factual
claims while precision/validity are N/A. A response with no claims receives N/A
groundedness, not an automatic perfect score. An empty model generation is an
execution error, not a correct abstention.

Two isolated judge tasks prevent hidden gold from supporting grounding labels.
The first sees the answer, actual evidence, tool records and parsed references;
the second sees verified reference facts and expected evidence limitations.
Quotations and identifiers are validated; scores are calculated from categorical
labels. Calibration tests correct, incorrect, uncited, fabricated-citation,
unsupported-but-correct, correctly abstaining, unnecessarily abstaining, incomplete
reference facts, contradictory-extra-claim, irrelevant-citation, partial-citation and jointly cited
answers. Calibration is a necessary check, not a claim of perfect agreement with
human review. Judge guidance follows
[OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices).

Aggregate scores are macro averages over scored cases. N/A, unscored and failed
counts are explicit, with category breakdowns and per-case judgments. Reports
record dataset/fixture/scorer hashes and versions, corpus SHA, application revision
and source hash, prompt hash, model/dependency settings, retrieval parameters, and
judge model, rubric, schemas, implementation, calibration and timeout settings.

## Comparing future architectures

Implement `Runner.run(question: str) -> EvaluationResult` for a new target. The
contract contains answer text, ordered final contexts, all code evidence consumed,
a reference catalog and tool records. Gold facts stay outside the runner.
The current adapter calls the real retriever and shared production generation
function; evaluation does not persist conversations or alter SSE behavior.

Future adapters can retain iterative evidence in `evidence_seen` while identifying
the actual final six code windows in `final_contexts`. Grounding uses consumed
evidence; retrieval metrics use the final windows. Record changed model and
retrieval budgets explicitly. The comparison command rejects different gold,
corpus, fixture, scorer or judge configurations, and reports paired metric deltas
while exposing intentional application/model changes. A single small benchmark
does not establish statistical significance.

## Verification

From `apps/api`:

```powershell
python scripts/uv_run.py pytest -p no:cacheprovider
python scripts/uv_run.py ruff check app tests
python scripts/uv_run.py mypy app
python scripts/uv_run.py python -m compileall -q app

$env:REPOMIND_EVAL_INTEGRATION = "1"
python scripts/uv_run.py pytest -p no:cacheprovider tests/test_evals_integration.py
```

The opt-in integration test uses a separate deterministic-embedding index identity,
the real production SQL and pgvector, and removes its own evaluation rows. It
verifies shared-connection serialization, idempotent seeding, parent retrieval and
corruption rejection. Those vectors demonstrate mechanics, not retrieval quality.
Only explicitly executed evaluations with real embeddings and generated answers
produce quality measurements.
