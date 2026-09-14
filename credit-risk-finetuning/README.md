# Fine-tuning workbench

The recommended entry point is the **native local dashboard** at `http://127.0.0.1:8090`. It has Datasets, Runs, Evaluation, and Answers & feedback views. Evaluation lists completed training runs as candidates and can queue a matched base/candidate pair with one action; training losses remain diagnostics rather than scorecards. No manual login or reviewer token is required. Supply datasets in phase 2; registering data or submitting feedback never starts training.

```sh
uv sync --frozen --extra dev --extra training --extra rag --extra documents --extra data
uv run --no-sync python -m credit_risk.workbench.server
```

Start with the [architecture, evaluation and continuous-learning guide](docs/architecture-and-learning-loop.md). Read the [phase-2 contract and operating guide](docs/workbench-data-contract.md) and [workbench validation report](docs/workbench-validation.md). Workbench training requires the V2 data contract. Feedback uses Submit, saves invalid corrections with diagnostics, and creates development checks only from independent expected results. Unknown causes remain untriaged. Prompt/schema recommendations leave active settings unchanged.

The following instructions describe the separate, existing SQL application. Its reviewer credentials do not apply to the fine-tuning dashboard.

## Existing SQL application

Python/uv development environment for synthetic or masked credit-risk factsheets, reviewed SQL, RAG, guardrails, evaluation, native MLX adapters and batch feedback. Advisory outputs require human review. Model promotion remains blocked until a reviewed benchmark and independent grounding/retrieval qualification pass. See the [mitigation and qualification guide](docs/mitigation-and-qualification.md) for changed contracts, commands, and release requirements.

The implementation and measured limits are recorded in [the validation report](docs/implementation-validation.md). The original 32-file handover is preserved separately in [the reference baseline](../references/handover-baseline).

## Run locally on Apple Silicon

```sh
uv sync --frozen --extra dev --extra rag --extra documents --extra training
uv run python scripts/setup_local.py
uv run credit-risk-data-prep fixture
docker compose up -d --build api data-service qdrant
```

Measure candidate JSONL before admission with the governed coverage matrix and shared
clone, template-family, split-leakage, and near-miss controls:

```sh
uv run credit-risk-data-prep coverage candidate.jsonl --out coverage.json
uv run credit-risk-data-prep diversity candidate.jsonl --out diversity.json
uv run credit-risk-data-prep rules rule-input.json --out rule-evaluations.json
```

These commands exit nonzero when checks fail. Coverage defaults to
`configs/data_prep/coverage_targets.yaml`; `--targets` selects another reviewed matrix.
The rules input contains a `factsheet` and its retrieved `evidence`; mandatory unevaluable
rules fail before generation.

`setup_local.py` creates private credentials only when neither `.env` nor `.reviewer-token` exists; it preserves existing configuration. Configure reviewers in `CR_REVIEWERS` as an identity map with `role` and SHA256 `token_sha256`. Never commit credentials. The service token and reviewer token serve different boundaries.

Open `http://127.0.0.1:8080/review`, enter the credential from `.reviewer-token`, prepare the example structured plan, inspect the masked SQL packet, approve, then execute. Submit changes through the structured plan correction action. Each correction requires a fresh approval; each approval allows one execution. Rejected, stale and replayed revisions fail closed. The page never executes edited SQL.

MLX training and oMLX serving remain native on macOS. Docker contains the API, internal read-only data service and Qdrant. Only the data service mounts `data/curated`; audit databases have separate writable mounts. The API uses native oMLX at `host.docker.internal:9905`. Configure an actual embedding model through `CR_EMBEDDING_MODEL` and pin it with `CR_EMBEDDING_REVISION`; either value being unset prevents API startup. Hashing is restricted to explicit offline tests (`CR_OFFLINE_TEST_MODE=true`). SAMA and CBUAE are separate knowledge domains. Reindex existing documents after the versioned-ID/ACL change.

## Validate

```sh
uv lock --check --offline
uv run --no-sync pytest -q
docker compose --profile test build tests
docker compose --profile test run --rm tests
docker run --rm --network credit-risk-finetuning_restricted-data \
  -e CR_TEST_QDRANT_URL=http://qdrant:6333 credit-risk-finetuning-tests \
  python -m pytest -q tests/test_qdrant_live.py
```

Live tests create and remove synthetic test collections. The offline suite intentionally skips services/model integrations that are not configured. Runtime health alone does not establish model or evidence quality.

## Build reviewed data and train a candidate

Each input JSONL record contains `case`, `question`, `evidence`, a JSON-string `target`, `task_type`, `data_classification`, and review metadata (`reviewer_id`, `status: approved`, `quality_score >= 4`). Cases require explicit `group_id`, obligor, portfolio, jurisdiction and as-of date. Unsupported targets are rejected. The same frozen exclusions must be supplied to seed and feedback pipelines.

```sh
uv run python -m credit_risk.dataset reviewed.jsonl data/training/v-next \
  --out-of-time-from 2026-01-01 --exclusions gold_exclusions.json
uv run python -m credit_risk.training --config configs/training.yaml
uv run python -m credit_risk.training --config configs/training.yaml --execute
uv run python -m credit_risk.training fuse --config configs/training.yaml \
  --save-path models/candidates/v-next --execute
```

Point the YAML at the new dataset manifest and a fresh `adapters/candidates/...` directory before running. Preflight prints metadata without training unless `--execute` is supplied. Two epochs derive micro-batches from actual training size and round to complete accumulation boundaries: with batch 1 and accumulation 8, 1,200 micro-batches mean approximately 150 optimizer updates, not 9,600 examples. Existing candidate/champion paths are protected.

For feedback, run `credit_risk.feedback INPUT OUTPUT --out-of-time-from DATE --exclusions FILE`. The optional Compose batch profile expects `data/training/gold_exclusions.json` and a fresh `candidate-batch` output; set `CR_OUT_OF_TIME_FROM` explicitly for the approved holdout. Review metadata, captured input/evidence and upstream group lineage are mandatory. SQL corrections remain audit/query feedback, not model SFT targets.

A mechanics-only 100-example spike has already completed under `outputs/spike-v3`; it used 24 micro-batches and successfully reloaded both adapter and fused candidate. It does not establish credit accuracy or full-context memory use. The synthetic generator refuses to overwrite that run. No 27B training is required or recommended at this stage.
