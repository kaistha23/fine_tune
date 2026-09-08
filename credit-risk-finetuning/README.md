# Credit-Risk Qwen Fine-Tuning Starter

Runnable development scaffold for a local, English-language credit-risk advisory copilot covering Retail, SME and Corporate portfolios under separate SAMA and CBUAE knowledge domains.

## What works today

Run `uv run python -m unittest discover -s tests -v` - 113 tests - and `docker compose config -q`.

| Capability | State |
|---|---|
| Default-deny schema registry: tables, columns, metrics, portfolios, grain, operators | Working |
| Default-deny architecture policy per service role | Working |
| Point-in-time query compilation, leakage prevented | Working |
| Parameterised SQL, obligor values never interpolated | Working |
| Restricted data service: read-only, memory/thread/time limits, result validation | Working |
| API/data-service separation, API has no database mount | Working |
| Human SQL review: packet, approve/reject, revalidated corrections, persisted feedback | Working |
| Deterministic ratios and the compact credit factsheet | Working |
| Feedback triage routing every root cause to an owner | Working |
| SFT dataset builder in mlx-lm chat format with provenance | Working |
| MLX-LM train and fuse commands | Command construction working; **unrun - needs the Mac** |
| RAG: per-jurisdiction collections, ACL and effective-date filters applied pre-search | Working |
| Hybrid dense + BM25 retrieval with reciprocal rank fusion | Working |
| Guarded inference: factsheet + evidence to a checked, cited answer | Working |
| Action control: risk tiers, prohibited autonomous decisions, abstention | Working |
| Release gates and champion-challenger promotion, scored per portfolio and task | Working |
| Document ingestion: heading/clause chunking with full lineage | Working |
| Embeddings | `MLXEmbedder` (Qwen3-Embedding) written; **retrieval quality still unmeasured**. `HashingEmbedder` remains the offline default and is not semantic |
| Qdrant backend | Working; server-side filter validated against a live v1.19 server to match the in-memory reference exactly |
| A real gold evaluation set | 8 synthetic seed cases, content-hashed. **Not a substitute for SME-written cases** |

## Endpoints

| Route | Purpose |
|---|---|
| `GET /health` | Status plus both pinned policy versions |
| `POST /v1/query/validate` | Validate a plan, return the SQL review packet. No database access |
| `POST /v1/query/review` | Record approve/reject; revalidate a corrected plan; persist feedback |
| `POST /v1/query/fetch` | Approved rows via the restricted data service (obligor, facility or suppressed portfolio cohort) |
| `POST /v1/factsheet` | Approved rows reduced to the compact factsheet the model is shown |
| `POST /v1/analyse` | The full guarded path: rows, factsheet, evidence, model, output check, action gate |

## Target architecture

- Qwen3.5-9B 4-bit common QLoRA adapter through MLX-LM.
- Qwen3.8-27B 4-bit as an on-demand teacher/challenger.
- oMLX for local inference evaluation.
- Dynamic portfolio schemas and parameterised query compilation.
- Deterministic Python/SQL for ratios and model outputs.
- Four guardrail layers: input, retrieval, output and abstention/action control.
- Feedback triage that sends errors to the correct component before considering retraining.

## Execution model: Docker plus native Mac services

The repository uses a hybrid architecture. This is intentional:

| Component | Execution | Reason |
|---|---|---|
| API gateway and guardrails | Docker container | Has no database mount and cannot execute SQL |
| Restricted data service | Docker container | Sole read-only database owner; compiles only allowlisted parameterised queries |
| Qdrant | Docker container | Vector store for SAMA/CBUAE namespaces. **Container only - no indexing or retrieval code yet** |
| Feedback batch worker | On-demand Docker container | Reads feedback and writes curated candidate batches only |
| Test runner | On-demand Docker container | Reproducible architecture and regression tests |
| oMLX | Native macOS process | Uses Apple Metal/ANE optimisations unavailable through Docker Desktop. **Client exists but is not yet wired to an endpoint** |
| MLX-LM QLoRA training | Native macOS process | Requires direct Apple Silicon acceleration and unified memory |

The API and data service are separated by an internal Docker network. Only the data service receives the read-only `data/curated` mount. The API receives validated rows, not SQL access. Both services load version-pinned, default-deny schema and architecture policies.

## Quick start

```bash
uv venv --python 3.12          # required: the repo pins >=3.12,<3.14
source .venv/bin/activate
uv sync --extra dev

# Synthetic, tokenised, deterministic. No real data anywhere in this repo.
uv run python scripts/make_fixture.py

uv run python -m unittest discover -s tests -v
```

Start the development API:

```bash
uv run credit-risk-api
```

## Docker quick start

Prerequisites:

- Docker Desktop.
- oMLX running natively on the host (see `CR_OMLX_BASE_URL`).
- A DuckDB file at `data/curated/credit_risk.duckdb` with the allowlisted tables.
  Generate a synthetic one with `uv run python scripts/make_fixture.py`.

```bash
cp .env.docker.example .env
docker compose build
docker compose up -d api data-service qdrant
docker compose ps
curl http://127.0.0.1:8080/health
```

Run the isolated test task:

```bash
docker compose --profile test run --rm tests
```

Run the feedback batch task after placing `feedback.jsonl` in `data/feedback/`:

```bash
docker compose --profile batch run --rm feedback-worker
```

Run QLoRA natively on macOS, outside Docker. Both commands print by default and only run
with `--execute`, so the config is always read before it is trusted:

```bash
uv sync --extra training
uv run credit-risk-train train --config configs/training.yaml --execute

# Fusing is not optional. oMLX serves model directories and has no adapter flag, so an
# unfused adapter cannot be served and therefore cannot be evaluated.
uv run credit-risk-train fuse --config configs/training.yaml --execute
```

`mlx_lm.server` can hot-load `--adapter-path` directly, which is faster while iterating; fuse
once the adapter is worth serving.

> **Before trusting a run:** `mlx-community/Qwen3.5-9B-4bit` is a vision-language checkpoint
> (`Qwen3_5ForConditionalGeneration`, with a vision tower). Confirm the installed mlx-lm loads
> it text-only, and do not route training through mlx-vlm - its Qwen3.5 LoRA path has an open
> p1 corruption bug. Fallback ladder is Qwen3 generation only: `Qwen3.8-27B-4bit` (the 64 GB
> challenger, borderline for QLoRA), then `Qwen3-14B-4bit`.

### Architecture-level schema enforcement

1. `configs/architecture_policy.yaml` defines explicit service capabilities with default deny.
2. `configs/schema_registry.yaml` allowlists tables, columns, metrics, portfolio applicability, grain, types and ranges.
3. The API validates the typed `QueryPlan` but has no database mount.
4. The restricted data service validates the same version-pinned schemas again.
5. The data service generates parameterised SQL internally, opens DuckDB read-only, and runs under a memory limit, a thread cap and a wall-clock bound.
6. Returned rows are re-validated for grain uniqueness, row limit, portfolio/jurisdiction consistency and the point-in-time bound before anything is built on them.
7. Docker isolates the data service on an internal network and mounts curated data as read-only. Only the API publishes a host port.
8. A designated human reviewer inspects the exact parameterised SQL template, its hash, schema version, grain, tables, columns, joins, filters and governed calculation ids. Parameter values are masked; the displayed SQL is read-only.

### Point-in-time correctness

Every `QueryPlan` carries a required `as_of_date`, and the compiler emits a mandatory
`data_cutoff_date <= as_of_date` predicate - plus `model_run_date <= as_of_date` whenever a
model-output column is selected, because a row can sit inside the data cutoff while its
PD/LGD/ECL came from a later model run. `date_to` may never exceed `as_of_date`. The data
service re-asserts both bounds on the rows that come back.

This is what stops future data leaking into a training example, where it would be invisible
downstream.

### Portfolio cohorts: aggregation without disclosure

An obligor plan compiles to `obligor_id = ?`. A portfolio plan compiles to a `GROUP BY`
over allowlisted dimensions - and may not name an obligor or facility at all, or a
per-obligor extract would be reachable by bolting a `GROUP BY` onto it.

Cohorts below 25 borrowers are suppressed in the `HAVING` clause and re-checked by the data
service on the rows returned. The floor counts `DISTINCT obligor_id`, not rows: the table is
obligor-month grain, so `COUNT(*)` over a year reaches 25 with three borrowers in it, which
would look like k-anonymity while providing none.

`MIN` and `MAX` are not available at any cohort size, because both return some individual's
actual value. Governed ratios cannot be aggregated in SQL either - the average of a ratio is
not the ratio of averages, and for DSCR the two differ by enough to change a decision, so
the compiler refuses instead of returning a plausible number.

### Retrieval: separation before search, not after

Each jurisdiction is a separate Qdrant collection, chosen from the jurisdiction before a
query is built - so a SAMA question never opens the CBUAE collection. Approval status,
confidentiality against the caller's role, the effective-date window and portfolio are all
**pre-search predicates** passed into the query, not filters applied to results.

That ordering is the control. Filtering after retrieval is not equivalent: by then the
document has already reached the process that builds the prompt. `validate_retrieval` still
runs afterwards, but as defence in depth - a cross-jurisdiction hit at that point means the
filter is broken, so it raises rather than quietly dropping the row.

Configured in `configs/retrieval.yaml`. An unknown role sees nothing; an unknown
jurisdiction has no collection.

### Action control

`POST /v1/analyse` gates every answer before release:

| Tier | Example | Outcome |
|---|---|---|
| Low | Factsheet, extraction, no recommendation | Auto-release |
| Medium | Deterioration commentary, any recommendation at all | Analyst review |
| High | SICR, stage migration, rating or limit change, covenant breach | Senior credit approval |
| Prohibited | "We hereby approve the facility", policy override | Blocked |

An unsupported or miscited claim is blocked regardless of tier, so a fluent answer with a
fabricated citation cannot be released. Abstention is a first-class outcome: with no
admissible evidence the route returns `INSUFFICIENT_EVIDENCE` and the factsheet, without
calling the model at all.

### Release gates

`configs/evaluation_thresholds.yaml` is read by `ReleaseGates` and applied to the overall
scorecard **and to every portfolio and task slice**, because an aggregate that passes while
one portfolio has collapsed is not a pass. Unsupported claims, cross-jurisdiction retrieval
and numerical disagreement are zero-tolerance.

A candidate adapter is promoted only if it clears every gate, improves something, and
regresses no portfolio. The champion is never overwritten - `compare_adapters` returns a
decision, it does not perform one.

### Human SQL review

`POST /v1/query/validate` returns the review packet. The reviewer approves or rejects through
`POST /v1/query/review`, quoting the `query_hash` they were shown - a stale hash is refused.

A correction is a **structured query plan**, never edited SQL. Corrections are recompiled
through the identical default-deny cycle, and one that breaks the rules is refused rather than
accepted. Every decision is persisted as a `FeedbackRecord` with `eligible_for_training=False`
and routed to the schema and query backlog: a query-layer defect is fixed in the query layer,
not by retraining the adapter.

SQL review feedback must identify the incorrect table, column, join, grain, filter or governed calculation. Any corrected query plan is sent through the complete schema and architecture validation cycle before execution. Database credentials, internal paths and raw database errors are never returned.

Validate a query plan:

```bash
curl -X POST http://127.0.0.1:8080/v1/query/validate \
  -H 'Content-Type: application/json' \
  -d '{
    "user_text": "Analyse the obligor credit deterioration",
    "query_plan": {
      "portfolio": "corporate",
      "jurisdiction": "SAMA",
      "entity_level": "obligor",
      "obligor_id": "OBL-0008",
      "date_from": "2025-01-31",
      "date_to": "2025-12-31",
      "as_of_date": "2026-01-15",
      "metrics": ["current_ratio", "dscr", "pit_pd", "stage"],
      "analysis_type": "credit_deterioration"
    }
  }'
```

The same request body works against `/v1/query/fetch` for approved rows, and against
`/v1/factsheet` for the compact factsheet built from them.

Record the reviewer's decision, quoting the `query_hash` the packet returned:

```bash
curl -X POST http://127.0.0.1:8080/v1/query/review   -H 'Content-Type: application/json'   -d '{
    "interaction_id": "INT-0001",
    "query_hash": "<query_hash from the review packet>",
    "reviewed_query_plan": { "...": "the same query_plan as above" },
    "decision": "rejected",
    "comment": "dscr missing from the deterioration view",
    "error_labels": ["wrong_column"],
    "root_cause": "schema",
    "corrected_query_plan": { "...": "a corrected plan, never edited SQL" }
  }'
```

A stale `query_hash` is refused with 409, and a correction that breaks the rules is refused
with 422 rather than accepted.

## Data safety

Raw, curated, training and model files are ignored by Git. Do not place confidential data in source control. Use tokenised identifiers in training data. The query compiler returns parameterised SQL and never interpolates obligor values into query text.

## Current boundary

Every stage of the pipeline now exists and is tested. What remains is not missing code but
missing real inputs and one unrun environment:

1. **No training run has completed on Apple Silicon.** The environment is confirmed - mlx
   0.32.2, mlx-lm 0.31.3 with `qwen3_5` among its supported architectures, Metal available -
   but the QLoRA run and `mlx_lm.fuse` have not been executed end to end. The commands
   construct correctly and are unit-tested. See the pre-flight check below.
2. **Retrieval quality is unmeasured.** `MLXEmbedder` runs Qwen3-Embedding through mlx-lm
   and is exercised by `tests/test_embedding_live.py`, but the corpus has not been indexed
   with it and no recall figure exists. `HashingEmbedder` is still the default: a hashed
   bag-of-words that matches on shared surface tokens only, and no accuracy claim may rest
   on it. Vectors from the two are not comparable, so switching requires a re-index - the
   index records the embedder signature and refuses a mismatch rather than scoring it.
3. **The gold set is synthetic.** Eight seed cases across both jurisdictions and all three
   portfolios, content-hashed so a case cannot drift unnoticed. They exercise the harness;
   they are not evidence of accuracy, and a credit SME still has to write the real ones.
4. **Only text ingestion exists.** `rag/ingest.py` chunks by heading and clause with full
   lineage, but PDF and Word extraction is not built - documents have to arrive as text.
5. **Queries are obligor-scoped only.** The compiler always emits `obligor_id = ?` and no
   joins, so portfolio-level cohort analysis is not reachable (finding M4).

The institution-specific PIT PD/ECL engines remain integration points: their physical schemas
and formulas must be supplied by the bank. The model never derives them - `model_outputs` in
the factsheet are reported as given.

### Verify before trusting a training run

Nothing in this repository has been executed on Apple Silicon. Before the first real run,
confirm on the Mac:

```bash
uv venv --python 3.12          # the Mac default is 3.14, outside requires-python
uv sync --extra training
python -c "import mlx_lm; print(mlx_lm.__version__)"
# then load mlx-community/Qwen3.5-9B-4bit text-only, run a 50-step spike, and fuse
```

If the load raises `Model type qwen3_5 not supported`, take the fallback ladder in
`configs/training.yaml` rather than building on an unproven base.
