# Credit-Risk Qwen Fine-Tuning Starter

Runnable development scaffold for a local, English-language credit-risk advisory copilot covering Retail, SME and Corporate portfolios under separate SAMA and CBUAE knowledge domains.

## Recommended architecture

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
| Qdrant | Docker container | Isolated vector store for SAMA/CBUAE namespaces |
| Feedback batch worker | On-demand Docker container | Reads feedback and writes curated candidate batches only |
| Test runner | On-demand Docker container | Reproducible architecture and regression tests |
| oMLX | Native macOS process | Uses Apple Metal/ANE optimisations unavailable through Docker Desktop |
| MLX-LM QLoRA training | Native macOS process | Requires direct Apple Silicon acceleration and unified memory |

The API and data service are separated by an internal Docker network. Only the data service receives the read-only `data/curated` mount. The API receives validated rows, not SQL access. Both services load version-pinned, default-deny schema and architecture policies.

## Quick start

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
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
> p1 corruption bug. Fallback ladder: `Qwen3-14B-4bit`, then `Qwen2.5-7B-Instruct-8bit`.

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

To retrieve approved rows through the isolated data service, change the endpoint to `/v1/query/fetch`. The request format is identical.

## Data safety

Raw, curated, training and model files are ignored by Git. Do not place confidential data in source control. Use tokenised identifiers in training data. The query compiler returns parameterised SQL and never interpolates obligor values into query text.

## Current boundary

This repository is a Phase 1 development scaffold. Document parsing, Qdrant indexing and the institution-specific PIT PD/ECL engines are integration points because their physical schemas and formulas must be supplied by the bank.
