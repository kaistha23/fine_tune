
cd credit-risk-finetuning
uv run --no-sync python -m credit_risk.workbench.server



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
interrupted job, or register a changed manifest automatically. Jobs that were still queued when the
workbench stopped are marked `interrupted` and must be started again. If either half of a paired
base/candidate comparison fails or is stopped, the other queued half is cancelled.

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
uv run credit-risk-data-prep fixture --out data/curated/credit_risk.duckdb
docker compose up -d data-service api
```

This refreshes the reviewed SQL application's synthetic source data. It does not change a
training dataset. If the refreshed source is used to construct training cases, create a new
phase-2 manifest with a new `dataset_version`, source snapshot hash, split checksums, and truthful
provenance.

**3a. Grow workbench source data over time.** The workbench keeps its own append-only copy of
the governed tables in `outputs/workbench/source/credit_risk.duckdb`; the Docker file above is never
modified. In **Datasets → Source data**, choose **Initialize from curated fixture** once (load 1).
For each new period, upload Parquet (preferred), CSV or JSONL records with the schema-registry
columns — one row per obligor-month or facility-month, without `load_id` — choose **Validate**,
fix any reported errors, then **Append as new load**. The same steps on the command line:

```sh
cd credit-risk-finetuning
uv run credit-risk-data-prep source-init
uv run credit-risk-data-prep fixture --month 2026-01 --out-dir data/incoming/2026-01
uv run credit-risk-data-prep load --table obligor_monthly --file data/incoming/2026-01/obligor_monthly-2026-01.parquet --dry-run
uv run credit-risk-data-prep load --table obligor_monthly --file data/incoming/2026-01/obligor_monthly-2026-01.parquet
```

Rows are never edited. A correction for an earlier period is a new row with the same grain and a
later `data_cutoff_date`; later snapshots use it, earlier snapshots and earlier as-of dates keep the
original. The same file cannot be loaded twice.

**3b. Add policy and regulation documents.** In **Datasets → Documents**, upload a PDF, DOCX,
Markdown or text file with its document ID, version, jurisdiction and effective-from date, then
**Register document**. Upload a revised policy as a new version (never replace a file); it takes
over on its effective date, and questions as at earlier dates still retrieve the older version.
Only `approved` documents are retrievable. Choose **Index documents** to embed new documents with
the cached Qwen3-Embedding model; the job appears in **Runs** and uses the single model lane.

**3c. Ask questions.** In **Ask**, type a question, choose the jurisdiction, portfolio, as-of date
and data snapshot (latest load by default), then **Draft plan with model** or **Write plan myself**.
Check the plan (obligor, dates, metrics) and choose **Run with this plan**. The first model step
starts a model session in the single model lane; it stays loaded for later questions and releases
after 10 idle minutes, on **Release GPU**, or when you start training or evaluation. Each answer
shows a badge: **New**, **Reused** (identical question and context, returned instantly),
**Unstable** (the three deterministic repeats disagreed; never reused), **Changed since the last
answer** (with the fields that changed and why), or a policy-rule abstention. Use **Give feedback on
this answer** to open it in Answers & feedback.

**3d. Close the loop.** On an Ask answer, **Confirm correct** or submit a correction in Answers &
feedback: identical questions then return that verified answer immediately. Use **Plan was wrong**
when you had to edit the model's plan. In **Datasets → Build dataset version from feedback**, pick a
registered V2 dataset and a new version name (optionally add reviewed paraphrases), build, then
**Register built dataset** and train it from **Runs**. In **Evaluation**, compare the new adapter
against the base or the previous adapter, and **Replay verified answers** so the consistency gates can
confirm earlier verified answers still hold.

**4. Collect and export workbench feedback.** In **Answers & feedback**, inspect a captured
answer (each is labelled base model, candidate, adapter, regression or imported, with its job),
submit a comment, and add a corrected JSON answer plus independent expected checks when
available. A correction is eligible for training only when it is schema-valid, classified as
`model_behaviour`, supported by its facts/evidence and review record, and outside every protected
validation/test/OOT group. Answers produced by the Evaluation page come from validation, test and
OOT cases, so their corrections become development checks only. To capture training feedback, run
an **Evaluate dataset** job on **Train only** from the Runs page.

Select the fragment task and choose **Download eligible feedback fragment**. When nothing is
eligible, the page lists skipped feedback by reason instead of downloading an empty file. The downloaded file is an input fragment, not a
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
5. After completion, open **Evaluation**, find the completed run under **Trained candidates**, keep
   `Best validation`, and choose **Evaluate and compare**. The workbench queues the unchanged base
   model and the trained adapter against the same validation/test/OOT cases. Select `Final` only
   when intentionally evaluating a completed run with no qualifying best checkpoint.
6. Follow both jobs under **Paired evaluations**. Scorecards appear automatically after both finish
   and keep validation, test and OOT separate. Promotion remains blocked until judge qualification,
   retrieval calibration, required benchmark coverage, all release gates, and the declared
   target-metric improvement pass.

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
uv run credit-risk-data-prep fixture --out data/curated/credit_risk.duckdb
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
