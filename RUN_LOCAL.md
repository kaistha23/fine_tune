# Run the credit-risk fine-tuning repository locally

This guide operates the governed **credit-risk advisory copilot** in
`credit-risk-finetuning/`. The repository-root rating demo is a separate project with separate
data and dependencies.

The local workbench and the Docker application are also separate:

- The workbench at <http://127.0.0.1:8090> owns the **Dataset registry**, training,
  evaluation, answers, and direct feedback. It runs as a native Python process and stores its
  state in `outputs/workbench/workbench.sqlite3`.
- Docker runs the reviewed SQL API, read-only data service, and Qdrant. It mounts
  `data/curated/credit_risk.duckdb`; it does not start the workbench or register training data.
- Native oMLX serves chat and embedding models to the Docker API. MLX training also runs
  natively on Apple Silicon.

## A. Everyday operation

**1. Start the workbench.** From the repository root:

```sh
cd credit-risk-finetuning
uv sync --frozen --extra dev --extra training --extra rag --extra documents --extra data
uv run --no-sync python -m credit_risk.workbench.server
```

Open <http://127.0.0.1:8090>. The workbench automatically reopens its local registry and run
history from `outputs/workbench/workbench.sqlite3`. It does not start training, resume an
interrupted job, or register a changed manifest automatically.

Use the **Datasets** view to confirm the intended immutable dataset version is registered. A
registered manifest is rechecked before preflight and again before a worker starts. Never edit a
registered manifest or one of its split files. Create and register a new version when data changes.

**2. Start the reviewed SQL application when you need it.** Ensure native oMLX is running and
has the configured chat and embedding models, then run:

```sh
docker compose up -d --build api data-service qdrant
docker compose ps
```

Open <http://127.0.0.1:8080/review> and use the token in `.reviewer-token`. Docker startup does
not populate the workbench Dataset registry. It only opens the existing local DuckDB fixture and
the persisted Qdrant volume.

**3. Refresh local SQL source data when needed.** Stop readers before replacing the DuckDB file:

```sh
docker compose stop api data-service
uv run python scripts/make_fixture.py --out data/curated/credit_risk.duckdb
docker compose up -d data-service api
```

This refreshes the reviewed SQL application's synthetic source data. It does not change a
training dataset. If the refreshed source is used to construct training cases, create a new
phase-2 manifest with a new `dataset_version`, source snapshot hash, split checksums, and truthful
provenance.

**4. Collect and export workbench feedback.** In **Answers & feedback**, inspect a captured
answer, submit a comment, and add a corrected JSON answer plus independent expected checks when
available. A correction is eligible for training only when it is schema-valid, classified as
`model_behaviour`, supported by its facts/evidence and review record, and outside every protected
validation/test/OOT group.

Choose **Download eligible feedback fragment**. The downloaded file is an input fragment, not a
registrable manifest. Keep its recommended maximum mixture: begin with no more than 20% validated
feedback and at least 80% curated anchor cases. Deduplicate it, retain explicit `group_id` values,
and never move validation, test, or OOT cases into training.

**5. Create and register a refreshed dataset version.** Make a new directory; do not modify the
old version:

```text
data/workbench/credit-analysis/2026-09-13.2/
├── manifest.json
├── train.jsonl       # anchor train cases plus eligible feedback
├── validation.jsonl  # frozen independent validation cases
├── test.jsonl        # frozen independent test cases
└── oot.jsonl         # frozen post-cutoff cases
```

Update every file checksum in `manifest.json`, advance `dataset_version` and `created_at`, and
record the new source/feedback lineage in each case's `provenance`. Validate the new manifest
locally before opening the UI:

```sh
uv run --no-sync python -c \
  'from credit_risk.workbench.contracts import inspect_dataset; import sys, pprint; pprint.pp(inspect_dataset(sys.argv[1]))' \
  "$PWD/data/workbench/credit-analysis/2026-09-13.2/manifest.json"
```

Then enter that absolute manifest path in **Datasets** and choose **Register dataset**.
Registration verifies checksums, required targets/expectations, explicit group and template-family
isolation, privacy-scan provenance for masked data, and OOT dates. Registration never starts a job.

**6. Preflight, retrain, and evaluate explicitly.** In **Runs**:

1. Select the task, newly registered dataset, cached 4-bit base model, and prompt/schema version.
2. Select **Train**, choose the bounded LoRA settings, and run **Preflight**.
3. Review token lengths, assistant-mask boundaries, split availability, and configuration.
4. Choose **Start job**. This is the action that starts native MLX training.
5. After completion, select **Evaluate dataset**, the candidate adapter, the `best` checkpoint,
   and validation/test/OOT splits. Use `final` only when intentionally evaluating a completed run
   that produced no qualifying best checkpoint.
6. Compare candidate and champion only on matching benchmark, prompt, evidence, generation, and
   scoring versions. Promotion remains blocked until judge qualification, retrieval calibration,
   required benchmark coverage, all release gates, and the declared target-metric improvement pass.

Feedback-specific development regressions can be run separately with **Regression**. They help
confirm a correction but do not count as independent release accuracy.

**7. Stop local processes.** Stop the workbench with `Ctrl+C`, then stop Docker if it was used:

```sh
docker compose down
```

`docker compose down` keeps the named Qdrant data volume. Do not add `--volumes` unless you intend
to discard that local index. The workbench registry remains in `outputs/workbench/`.

## B. First-time setup with local data

**1. Install prerequisites.** Use Python 3.12 or 3.13 and `uv`. Apple Silicon is required for
native MLX training. Docker Desktop is required only for the reviewed SQL application.

```sh
git clone <repository-url> fine_tune
cd fine_tune/credit-risk-finetuning
uv sync --frozen --extra dev --extra training --extra rag --extra documents --extra data
uv lock --check
uv run --no-sync pytest -q
```

**2. Create private Docker configuration and synthetic SQL data.** Run this once:

```sh
uv run python scripts/setup_local.py
uv run python scripts/make_fixture.py --out data/curated/credit_risk.duckdb
```

**3. Create the phase-2 Dataset registry files.** Run:

```sh
uv run python scripts/make_workbench_fixture.py \
  --source data/curated/credit_risk.duckdb \
  --out data/workbench/credit-analysis/2026-09-13.1 \
  --version 2026-09-13.1
```

Check the generated files:

```sh
ls -lh data/workbench/credit-analysis/2026-09-13.1
cat data/workbench/credit-analysis/2026-09-13.1/manifest.json
```

Validate the manifest:

```sh
uv run --no-sync python -c \
  'from credit_risk.workbench.contracts import inspect_dataset; import pprint; pprint.pp(inspect_dataset("data/workbench/credit-analysis/2026-09-13.1/manifest.json"))'
```

Print the absolute manifest path:

```sh
python3 -c 'from pathlib import Path; print(Path("data/workbench/credit-analysis/2026-09-13.1/manifest.json").resolve())'
```

**4. Start and populate the workbench registry.** Run:

```sh
curl --max-time 2 -fsS http://127.0.0.1:8090/api/session
```

If that prints session JSON, open <http://127.0.0.1:8090> and do not start another server.

If it times out or reports a connection error, clear any suspended workbench process:

```sh
WORKBENCH_PIDS=$(lsof -t outputs/workbench/scheduler.lock 2>/dev/null | sort -u)
if [ -n "$WORKBENCH_PIDS" ]; then
  kill $WORKBENCH_PIDS
  kill -CONT $WORKBENCH_PIDS 2>/dev/null || true
fi
```

Start the server in the background:

```sh
mkdir -p outputs/workbench
nohup uv run --no-sync python -m credit_risk.workbench.server \
  > outputs/workbench/server.log 2>&1 < /dev/null &
sleep 2
curl --max-time 2 -fsS http://127.0.0.1:8090/api/session
```

Follow the log with:

```sh
tail -f outputs/workbench/server.log
```

Open <http://127.0.0.1:8090>.

1. Open **Datasets**.
2. Paste either the dataset directory or manifest path:

   ```text
   /Users/gklab/repos/fine_tune/credit-risk-finetuning/data/workbench/credit-analysis/2026-09-13.1
   /Users/gklab/repos/fine_tune/credit-risk-finetuning/data/workbench/credit-analysis/2026-09-13.1/manifest.json
   ```

3. Select **Register dataset**.
4. Open **Runs**.
5. Select `credit_analysis` and the registered dataset.
6. Select the cached base model and prompt/schema version.
7. Select **Train**.
8. Select **Run preflight**.
9. Select **Start job** after preflight passes.

**5. Configure and start Docker.** List the oMLX models:

```sh
curl -s http://127.0.0.1:9905/v1/models | uv run python -m json.tool
```

Edit `.env` and set these values:

```dotenv
CR_EMBEDDING_MODEL=<embedding model id from oMLX>
CR_EMBEDDING_REVISION=<embedding model revision or checksum>
CR_OMLX_API_KEY=<oMLX API key, or leave empty>
```

Start the services:

```sh
cd fine_tune/credit-risk-finetuning
docker compose up -d --build api data-service qdrant
docker compose ps
```

Open <http://127.0.0.1:8080/review>. Use the token printed by:

```sh
cat .reviewer-token
```

Continue with **A. Everyday operation** for refresh, feedback, retraining, and evaluation.
