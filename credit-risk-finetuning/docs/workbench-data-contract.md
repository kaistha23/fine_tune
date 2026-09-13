# Phase-2 data contract and local dashboard

The workbench prepares infrastructure only. Supply your own datasets in phase 2. Unit/browser fixtures are temporary test inputs, not credit training data or accuracy evidence.

## Start

From `credit-risk-finetuning/`:

```sh
uv sync --frozen --extra dev --extra training --extra rag --extra documents --extra data
uv run --no-sync python -m credit_risk.workbench.server
```

Open `http://127.0.0.1:8090`. No reviewer token, Docker stack or manual login is needed. The app binds to loopback, checks browser origins and automatically supplies an anti-CSRF token. This token is browser plumbing, not a credential you enter. The pre-existing SQL gateway is a separate application.

State lives in `outputs/workbench/workbench.sqlite3`; run specs/logs/results live under `outputs/workbench/runs/`. Training outputs use `adapters/candidates/<task>/<run-id>/`. A single scheduler owns the workspace; all workbench training/evaluation workers share a native-process lock. This does not coordinate unrelated training commands outside the workbench. An orphaned worker retains the lock until it exits; restarted runs are marked interrupted and are never automatically resumed.

## Dataset manifest

Register a local manifest path from the Datasets view. Required shape:

```json
{
  "format": "credit-workbench-v2",
  "dataset_id": "credit-analysis-synthetic",
  "dataset_version": "2026-09-09.1",
  "created_at": "2026-09-09T00:00:00Z",
  "name": "Your phase-2 dataset version",
  "task": "credit_analysis",
  "oot_from": "2026-01-01",
  "splits": {
    "train": {"file": "train.jsonl", "sha256": "<file SHA256>"},
    "validation": {"file": "validation.jsonl", "sha256": "<file SHA256>"},
    "test": {"file": "test.jsonl", "sha256": "<file SHA256>"},
    "oot": {"file": "oot.jsonl", "sha256": "<file SHA256>"}
  }
}
```

All four files must be declared; empty files are permitted for registration. V2 train and validation cases require targets, and validation/test/OOT cases require independent expected checks. Training needs nonempty train and validation files. Paths must stay within the manifest directory, including resolved symlinks. Checksums are verified at registration, preflight and worker execution. Do not overwrite a registered version. V1 manifests remain inspectable for migration but cannot start workbench training.

Masked V2 data also requires `privacy_scan: {"status":"passed","scanner_version":"..."}`. This is provenance for the upstream scan, not a claim that the workbench discovered every sensitive value.

Use separate manifests/adapters for `credit_analysis` and `query_plan`. A query-plan manifest additionally requires `schema_registry: {"file": "schema_registry.yaml", "sha256": "..."}` in the repository's governed registry format. Supply the relevant schema context in each question's `facts`; the model cannot infer schema information it was not shown. Registry formulas must be supported by the deterministic compiler/calculators.

For execution-based query metrics, also provide:

```json
"synthetic_snapshot": {
  "file": "synthetic.duckdb",
  "sha256": "<database SHA256>",
  "classification": "synthetic"
}
```

Only declared synthetic snapshots are executed. Execution uses compiled QueryPlan objects, read-only DuckDB, resource limits, result validation and before/after file hashes. Expected results alone do not authorize raw SQL execution. Without a snapshot and expected rows, result agreement is **Not evaluated**.

## Each JSONL case

Required fields: `case_id`, `group_id`, `task`, `split`, `question`, `portfolio` (`retail`, `sme`, `corporate`), `jurisdiction` (`SAMA`, `CBUAE`), `as_of_date` (ISO date), and `provenance.classification` (`synthetic` or `masked`). V2 additionally requires `provenance.source_snapshot_hash`, `provenance.template_family`, `provenance.transformations`, and an explicit `fact_records` array for credit cases. This declaration does not certify regulatory correctness.

Optional/phase-dependent fields:

| Field | Meaning |
|---|---|
| `facts` | Exact model factsheet or query-schema context; preserve numerical types and case identity. |
| `fact_records` | Typed facts with `fact_id`, `metric`, `value`, `unit`, `currency`, `effective_date` and `source_id`. Use stable source IDs and retain the governed factsheet in `facts`. |
| `evidence` | Evidence objects in the existing Evidence schema, including versioned IDs and text. |
| `target` | JSON object matching the selected output schema; required for train/validation preflight. |
| `expected` | Independent checks, described below. Absent checks do not produce perfect scores. |
| `consistency_paths` | Dot paths into stable JSON fields, e.g. `answer_status`, `risk_drivers`, or an explicit numeric/conclusion field in a compatible schema extension. |
| `equivalence_id` | Explicitly declared meaning-preserving question family. Context, expectations and split must match. |
| `distinct_from` | Other case IDs expected to produce different checked projections; use for known negative controls, not merely different wording. |
| `sql_lineage` | Optional source review packet for imported facts; no query is executed from this field. |
| `provenance.template_family` | Template family ID; prevent that family crossing splits. |

Borrower groups and declared template families cannot cross splits. Train/validation/test dates must precede `oot_from`; OOT dates must be on or after it. Case IDs and exact question/context pairs must be unique. Registration also checks protected groups across registered datasets. These controls depend on truthful upstream group/family metadata; they do not detect all paraphrase leakage automatically.

### Expected checks

`expected` supports:

- `fields`: mapping of dot paths to expected values. Used for statuses and structured conclusions. List order and harmless string whitespace/case are normalized; prose entailment is not inferred.
- `numerics`: mapping of dot paths to either a number or `{value, abs_tolerance?, rel_tolerance?, unit?, unit_path?, currency?, currency_path?, as_of_date?, as_of_date_path?}`. Missing, non-finite, wrong-unit and wrong-date outputs fail.
- `risk_drivers`: canonical driver labels for precision/recall. Use consistent labels across paraphrases.
- `evidence_ids`: required citations for citation recall.
- `must_abstain: true`: contributes to the abstention-recall denominator.
- `confidence_labels`: maps confidence-field paths to independently reviewed binary outcomes for Brier scoring. Do not interpret model confidence as calibrated until enough held-out labels exist.
- `severity_weight`: positive case weight for the business-weighted pass rate.
- `query_plan`: expected QueryPlan object. Compare normalized fields, not SQL wording.
- `tables`, `columns`, `joins`: expected compiler metadata.
- `rows`: expected JSON-safe result rows, order independent, duplicate multiplicity preserved; requires the synthetic snapshot.

All initial credit outputs retain the CreditResponse contract. The recommended V2 schema adds typed conclusions, risk-driver details, missing-information details and a structured recommendation while retaining the existing fields. Existing workspaces receive this as an inactive version; it is never selected automatically. Prompt/schema versions may add compatible fields; changing core required fields needs a separately implemented migration. Saved incompatible versions remain inspectable but cannot be activated or used for a run. Existing development targets must validate against a newly selected schema. No test/OOT labels are used for version compatibility or checkpoint selection. Prompt schemas omit titles, descriptions and defaults to reduce constant token overhead without changing validation rules.

## Evaluation and feedback

The deterministic profile repeats each selected question three times with native greedy decoding, seed sequence 42/43/44, output budget 1,024 tokens and explicit `enable_thinking=false`. The serving-consistency profile uses the same three seeds with temperature 0.1, top-p 0.9 and top-k 20, matching the local structured-output client. The context budget is at least 4,096 and follows the selected bounded sequence setting. Record these settings, model snapshot, adapter hash, prompt/schema hashes and selected checkpoint. Repeated results are a consistency diagnostic, not three independent accuracy observations or a universal determinism guarantee across hardware/library revisions.

Consistency compares the declared stable fields, not narrative wording. Query-plan consistency uses normalized QueryPlan fields. Missing projections fail a requested consistency check; no configured credit projection is **Not evaluated**. Equivalent families require identical facts/evidence. Negative controls are evaluated only when their referenced cases are in the selected evaluation split set.

Model/adapter comparisons require identical cases, evaluator, prompt/schema version and generation settings. The separately labeled Prompt experiment mode permits prompt changes while retaining the same model/checkpoint/output schema and generation settings. Incompatible schema comparisons are not silently coerced.

Credit grounding is a **conservative extractive heuristic**, separate from citation resolution and numerical checks. It may reject valid paraphrases; it does not establish expert credit judgment. Missing labels show Not evaluated. Provider errors and invalid output remain failed cases rather than being dropped.

Metric rows include deterministic bootstrap 95% intervals and mark denominators below ten as small samples. Model comparisons also report paired per-case deltas, wins, losses and ties. If `confidence_labels` are present, evaluation reports Brier score, expected calibration error and reliability bins; confidence remains uncalibrated when those labels are absent or insufficient.

Feedback has a single Submit action. Comments need no root cause or correction. Invalid corrections are retained with diagnostics. `unknown` remains internally `untriaged`; it is never silently converted to a model defect. A correction cannot create its own expected result. Only a schema-valid `model_behaviour` correction with independent testable expectations from an unprotected development/train group can enter the downloadable batch fragment. Validation/test/OOT groups cannot. Every submission is preserved; retries use `submission_id` for idempotency and only the latest correction per interaction contributes to a future batch. For API clients, reuse the same submission ID on transport retries.

A corrected answer produces a development regression only from separately supplied checks or existing independent case checks. These checks are labeled development, including whether feedback can enter training; do not count them as independent accuracy. A comment or correction without expectations is saved as needing expected results. Active prompts/models remain unchanged.

The feedback export recommends a starting mixture of 20% validated feedback and 80% curated anchor examples. It does not perform that merge. Deduplicate and balance the merged V2 dataset, then rerun all split and checksum controls.

## Bounded training search

The Runs view exposes the supported local experiment surface: batch/accumulation, sequence limit, rank, scale, dropout, 1/4/8/16/32 adapted layers, explicit attention/MLP presets, Adam or AdamW, weight decay, constant or cosine schedule, warm-up, minimum learning-rate ratio, seed, patience and minimum loss improvement. The safe Qwen3.5-9B local default is one attention-only layer at rank 8; increase capacity only after a native memory check. Invalid combinations fail preflight. Start with learning rates `1e-5`, `2e-5`, `5e-5`; test adapter capacity only after choosing a learning-rate region, and rerun finalists with multiple seeds. Test and OOT remain outside selection.

The cached checkpoint is recorded as MLX quantized LoRA with its actual 4-bit affine/group-size metadata. No NF4, double-quantization or paged-optimizer claim is inferred from the term QLoRA.

Recommendations are deterministic checklist suggestions, **not an automatic diagnosis or a measured improvement**. The version editor saves a new prompt/schema directly; Use version explicitly selects it for new runs and can restore an older version. There is no approve/accept/reject feedback workflow. Exported feedback is a versioned input fragment; merge it into a new phase-2 dataset and rerun split/target validation before explicitly starting training.

## Local interfaces

- `GET /api/session`, `GET /api/state`: automatic browser session token and workspace state.
- `POST /api/datasets {path}`: register a manifest.
- `POST /api/preflight`, `POST /api/jobs`: task, kind (`train`, `evaluate`, `regression`), dataset/version/model IDs, checkpoint and allowed training settings.
- `GET /api/jobs/{id}`, `POST /api/jobs/{id}/stop`: artifacts and explicit stop.
- `GET /api/compare/{left}/{right}?mode=model|prompt`: compatible result comparison.
- `POST /api/answers`: import `{case, output, version_id, identity?, sql_lineage?}`. Imported answers are labeled as imported, not measured native runs.
- `POST /api/feedback`: `{submission_id, interaction_id, comment?, correction?, cause?, expectations?}`. No approval fields.
- `GET /api/feedback/batch/{task}`: latest eligible correction fragment.
- `POST /api/versions {task,prompt,schema,name,parent?}`, `POST /api/versions/{id}/activate`: version history and explicit selection/rollback.

State-changing requests require the automatically supplied `X-Workbench-Token` header and same-origin browser access. APIs do not accept arbitrary commands or executable SQL. Submission and job specs are persisted locally; treat the workspace as private development state.
