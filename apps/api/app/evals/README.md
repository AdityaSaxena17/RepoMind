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

The earlier child-chunk Qwen reranker trial scored Recall@6 **75.00%** and Precision@6
**34.31%**, so it remains opt-in. Its 24-answer full run has **18 accepted semantic
judgments**, with six failures remaining after API-limited retries. These partial
semantic results are not a complete quality baseline; see [the experiment](EXPERIMENTS.md).

Reranking the expanded parent windows instead scored Recall@6 **82.35%** and
Precision@6 **46.08%** across the same 17 applicable questions. All 24 retrieval
cases completed. This improves both macro scores over child reranking and the
no-reranker baseline; the latter recovers the same 31/40 evidence units. Parent
reranking remains opt-in, and its generation/semantic metrics have not been run.

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

Preparation uses local Qwen embeddings through Ollama by default. Full generation uses
`HIVE_API_KEY` when configured, with `HIVE_CHAT_MODEL` defaulting to
`deepseek-ai/deepseek-v4.1-flash`. Otherwise the existing OpenAI chat path requires
`OPENAI_API_KEY` and `OPENAI_CHAT_MODEL`. Cached vectors are validated for
dimensions and finite values. Corpus identities include source hashes, embedding model/dimensions and
actual chunk metadata; changed settings create a separate evaluation index.

## Environment files

Keep Hive, embedding, reranker, Redis and evaluation-database settings in
`apps/api/.env`. Keep Clerk, the web database URL, `FASTAPI_BASE_URL` and the web
GitHub token in `apps/web/.env`. The API also uses a GitHub token for indexing
and MCP calls; its template documents that setting explicitly.

API environment precedence is: explicit process variables, `apps/api/.env`,
`apps/web/.env`, then root `.env`. The web/root files remain fallbacks for
existing setups. Explicit shell or CI variables override file defaults.
`INTERNAL_API_CREDENTIAL` is optional locally; when configured, the web and API
values must match. Restart running processes after changing their environment.

## Commands

### Local Qwen embeddings

The default embedding model is `qwen3-embedding:0.6b` through Ollama, with native
1,024-dimensional vectors and pinned Q8_0 model digest
`ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`.
No API key or additional Python dependency is needed for embeddings.

```powershell
ollama pull qwen3-embedding:0.6b
# In apps/api/.env:
# EMBEDDINGS_PROVIDER=ollama
# OLLAMA_BASE_URL=http://127.0.0.1:11434
pnpm --filter api eval migrate
pnpm --filter api eval prepare
pnpm --filter api eval retrieval --no-rerank --experiment qwen-embedding-no-reranker --timeout 300
```

Queries receive the instruction `Given a question about a code repository,
retrieve relevant code passages that answer the question.` Source chunks are
embedded without an instruction. Batches are bounded at 64 texts, inputs are
never silently truncated, and the configured context limit is 8,192 tokens.
The model supports 32,768, but reserving that full window caused CPU offloading
on the local 6 GB GPU. The smaller allocation still exceeds the existing
512-token code chunks; oversized inputs fail explicitly rather than truncate.
Requests verify the local model digest and reject invalid counts, dimensions,
non-finite values and zero vectors. Ollama errors propagate; there is no paid
embedding fallback.

The migration preserves existing 1,536-dimensional OpenAI vectors and supports
native 1,024-dimensional vectors in the same pgvector column with separate
dimension-specific HNSW indexes. Each newly indexed chunk records its embedding
profile. Vector search rejects missing or mismatched profiles before embedding
the query, so existing application repositories need reindexing after migration.
Apply production migrations with the project's normal Prisma deploy command,
then use the existing repository reindex action. Evaluation migration/seeding
uses only the dedicated `repomind_eval` database.

Qwen corpus IDs and embedding caches include the provider, model digest,
dimensions, query instruction and context/truncation settings. Historical OpenAI
cache entries remain intact. Select `EMBEDDINGS_PROVIDER=openai` for a paired
comparison; regenerate the index profile with `eval prepare` if needed.
Dataset annotations, scorer, filtering, chunking, FTS, RRF, parent windows and
reranking are unchanged by the embedding switch. Embedding settings are recorded
in every new run. See [Ollama's embedding API](https://docs.ollama.com/api/embed),
[Qwen's model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), and
[pgvector's dimension-specific indexing](https://github.com/pgvector/pgvector#can-i-store-vectors-with-different-dimensions-in-the-same-column).

### Optional local Qwen reranker

The `rerank` extra installs Sentence Transformers 6.1.0 and PyTorch 2.10.0.
On Windows x64, uv uses PyTorch's official CUDA 12.8 wheel so a compatible NVIDIA
GPU can be used; this is a larger download than the CPU wheel. Other platforms
use PyPI. Scoring automatically uses CUDA when available, otherwise CPU.

From `apps/api`, run the same production retriever with reranking enabled:

```powershell
python scripts/uv_run.py --extra rerank python -m app.evals retrieval --rerank --experiment qwen-retrieval --timeout 300
python scripts/uv_run.py --extra rerank python -m app.evals full --rerank --experiment qwen-hive-full --timeout 300 --judge-model deepseek-ai/deepseek-v4.1-flash --judge-reasoning-effort high --judge-timeout 120
```

Weights download on the first enabled run into the ignored root `.model-cache/`.
The official model revision is pinned; no GGUF, Ollama, or remote model code is
used. Reports record revision, instruction, device, dtype, length limit, batch
size, input format, scoring placement, and dependency versions. Once cached,
`HF_HUB_OFFLINE=1` avoids model-download network requests while retaining
generation/judge access to the configured provider. If Xet transfer stalls during
initial download, Hugging Face supports `HF_HUB_DISABLE_XET=1` to use HTTP transfer instead.

The reranker now scores expanded parent windows rather than child chunks. The
unchanged vector/keyword search and RRF produce the full candidate pool; parent
expansion and merging process that entire pool before deduplicating visible
windows. Qwen scores each window's path and complete visible text with trained
yes-minus-no logits, and the highest six become the final contexts. Ties retain
the parent order derived from RRF. Source indices are assigned after selection.

Parent size limits remain 250 lines and 16,000 characters. Tokenizer truncation
is disabled; inputs exceeding the model's 32,768-token limit fail explicitly
instead of silently scoring a different excerpt. KV caching is disabled for
the single forward pass to reduce GPU memory use. There are no relevance
thresholds or changed embedding/index requirements. Model errors surface as
failed cases; they never silently fall back to RRF.

Production remains opt-in through `REPOSITORY_RERANKER=qwen3` and launching the
API with `python scripts/uv_run.py --extra rerank uvicorn app.main:app --host 127.0.0.1 --port 8137`.
The active setting is `REPOSITORY_RERANKER=none` to avoid the extra reranking
latency. Its implementation is retained; a commented `REPOSITORY_RERANKER=qwen3`
line in the environment template makes the opt-in easy to restore. Evaluation
`--rerank` and `--no-rerank` override that
setting for paired experiments. Unit tests never load or download model weights;
the API type-check command installs the optional package to check its types.

### Existing quality commands

#### Hive-hosted DeepSeek evaluation judge

Set `HIVE_API_KEY` in the ignored `apps/api/.env` to your **Hive** API key,
matching the working Hive endpoint. Select `deepseek-ai/deepseek-v4.1-flash`:

```powershell
pnpm --filter api eval calibrate --judge-model deepseek-ai/deepseek-v4.1-flash --reasoning-effort high --timeout 120 --output ../../.eval-artifacts/hive-deepseek-high-calibration-v2.json
pnpm --filter api eval grade ../../.eval-artifacts/single-agent-v1-accepted.json --judge-model deepseek-ai/deepseek-v4.1-flash --judge-reasoning-effort high --judge-timeout 120 --output ../../.eval-artifacts/single-agent-v1-hive-deepseek-high-v2.json
```

This judge uses `openai.AsyncOpenAI` with `https://api.thehive.ai/api/v3` and
defaults to high reasoning. The OpenAI SDK is already installed through
LangChain and is now declared as a direct dependency; no additional inference
framework is required. Production answer generation continues through LangChain,
using the same `HIVE_API_KEY` at the same Hive endpoint. It uses Chat Completions
streaming, high reasoning, a 16,384-token completion cap, an 8,192-token thinking
budget, a 120-second request timeout and one provider retry. `HIVE_CHAT_MODEL`
can select another Hive model; its default is `deepseek-ai/deepseek-v4.1-flash`.
Hive takes precedence over legacy OpenAI chat settings when its key is present;
provider errors surface without an automatic OpenAI fallback.

Selecting a GPT judge explicitly retains its existing OpenAI implementation and
requires OpenAI credits. For Hive judging, pass the model shown above; grading
remains opt-in. Hive uses only `HIVE_API_KEY`. The old `DEEPSEEK_API_KEY` name is
no longer read. Restart running API/worker processes after renaming the key.

Embedding calls use local Ollama, independently of the Hive chat credential.
`EMBEDDINGS_PROVIDER=openai` explicitly selects the historical embedding provider
and needs `OPENAI_API_KEY` for cache misses.

Hive's non-streaming JSON mode was verified with a live high-reasoning protocol
probe. The two judge tasks retain the same rubric, gold/evidence separation and
scorer. The full JSON response is validated against the existing Pydantic schema;
empty, malformed or truncated responses fail. Exact quotations and evidence IDs
still need to pass validation. All 12 calibration examples must pass before
grading. Requests have bounded timeouts, one provider retry, a 16,384-token
completion cap and an 8,192-token thinking budget. The explicit thinking budget
reserves room for the JSON labels while retaining high reasoning. HTTP clients
close on both success and failure.

Reports record Hive's provider/endpoint, SDK version, reasoning effort, output
format/budget and provider implementation hash. Regrade both compared answer sets
with matching judge configurations. Original GPT artifacts remain separate.
The strict `judge-agreement` command also requires matching grader implementation
hashes. Historical GPT artifacts predate the provider-call extraction, so they
cannot be used for that command without regrading GPT under the current code.
The historical GPT baseline is preserved; no cross-implementation agreement is
claimed.

Hive publishes its own prices; DeepSeek's direct-API peak/off-peak schedule does
not establish Hive's billing. On 2026-10-14, Hive lists discounted rates of
$0.12/M input tokens, $0.48/M output and $0.0024/M cache reads. Check
[Hive pricing](https://thehive.ai/pricing) for future runs. Integration follows
[Hive's model documentation](https://docs.thehive.ai/docs/deepseek-v41-flash).

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
pnpm --filter api eval calibrate --judge-model deepseek-ai/deepseek-v4.1-flash --reasoning-effort high --timeout 120

# Capture production-equivalent answers and grade with explicit opt-in.
pnpm --filter api eval full --experiment hive-full --judge-model deepseek-ai/deepseek-v4.1-flash --judge-reasoning-effort high --judge-timeout 120

# Capture answers without semantic grading.
pnpm --filter api eval full --experiment capture

# Grade saved answers without regenerating them.
pnpm --filter api eval grade ../../.eval-artifacts/capture.json --judge-model deepseek-ai/deepseek-v4.1-flash --judge-reasoning-effort high --judge-timeout 120 --output ../../.eval-artifacts/capture-graded.json

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

## Historical maximum scores for reporting

[MAXIMUM_SCORES.md](MAXIMUM_SCORES.md) preserves the user-selected historical
best-results table: completed reranked retrieval scores and the earlier partial
semantic readings. [data/maximum-scores-v1.json](data/maximum-scores-v1.json)
records source hashes, configurations, scored-case contributions and coverage.
These scores come from separate experiments; the semantic run accepted 18/24
judgments. The document includes report wording that keeps this coverage clear.
The current runtime configuration and baseline reports are unchanged.

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
