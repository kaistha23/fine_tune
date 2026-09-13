# Correctness fixes and release qualification

The implementation separates training admission, serving acceptance, and independent semantic evaluation. Promotion remains blocked until reviewed judge/retrieval qualification, benchmark coverage, and lineage checks succeed. The included review template is **unreviewed synthetic example data**, not qualification evidence. Group lineage sourcing remains deferred; no obligor fallback or historical factsheet migration is performed.

## Changed contracts

- Governed schema registry **1.3.0** and five ratio formula IDs ending in **`.v2`**. Nonpositive denominators and nonfinite numbers yield an invalid null value. Meaningful negative numerators remain valid. Tiny positive denominators have no invented business cutoff; overflow is invalid. Invalid metrics appear in factsheet quality flags.
- Dataset format **v4.0.0** retains explicit group IDs. Feedback reports `rejected_missing_group_id`. Factsheet schemas remain unchanged.
- Training completion manifest **v2** records baseline loss, selected loss/iteration/optimizer updates, and best/final checkpoint hashes. “Best” requires a post-update improvement and has no final-checkpoint fallback. Old best checkpoints cannot be exported as newly verified best checkpoints; a successful historical run can still be explicitly exported with `--checkpoint final`.
- Ingestion emits **v2** chunk/citation identities containing the full heading-path identity and repeated-heading occurrence. Readable section labels remain on evidence. Old collections and historical citations are preserved.
- Evaluation reports **v2** replace `faithfulness` with `extractive_support_rate`. Historical reports must be rerun for comparisons; do not relabel old results. The CLI seals reports with a content hash and refuses to overwrite them.

## Deployment and indexing

Set `CR_EMBEDDING_MODEL` to the model actually served by native oMLX, plus `CR_OMLX_API_KEY` where required. Update existing environments to `CR_SCHEMA_REGISTRY_VERSION=1.3.0`. Unknown `CR_` keys are errors. The documented `CR_TEST_*` integration-test keys remain supported.

Hashing requires explicit `CR_OFFLINE_TEST_MODE=true`; never set that on a deployment or a qualification run. `/health` reports offline-test mode distinctly, and in normal operation returns 503 when embeddings or configured Qdrant collections/signatures are unavailable. Health checks are availability checks, not evidence-quality certification.

Prepare a JSONL source file with one `{ "text": "...", "meta": { "document_id": "...", "jurisdiction": "SAMA", "document_version": "...", "approval_status": "approved", ... } }` per document. Use all governed source documents, including lifecycle/ACL metadata. From the project directory:

```sh
uv run credit-risk-reindex --documents governed-documents.jsonl \
  --suffix reviewed_20260911 --out outputs/reindex-20260911
```

This creates entirely new collections, verifies every citation payload and document chunk count, and writes a manifest and `retrieval.yaml`. It does not delete or switch existing collections. Deploy the verified policy using `CR_RETRIEVAL_POLICY` (and mount its path into the API container). Roll back by restoring the previous policy; previous namespace names are recorded in the manifest. A failed reindex does not generate a switchable policy; retry using a fresh suffix after investigating the failed collections.

Both retrieval arms generate candidates. All candidates pass the same absolute cosine threshold and access checks before the top-k/context budget is filled. RRF ranks results only; its values are not support probabilities.

## Reviewed training paraphrases

Exact supported extracts retain the conservative admission path. A non-extractive target additionally requires a `semantic_review` object:

```json
{
  "reviewer_id": "actual-reviewer-id",
  "status": "approved",
  "semantic_supported": true,
  "numerics_verified": true,
  "content_hash": "hash-of-the-exact-reviewed-input"
}
```

Compute `content_hash` with `credit_risk.review_store.digest` over `{ "target": CreditResponse.model_dump(mode="json"), "factsheet": case, "evidence": [Evidence.model_dump(mode="json"), ...] }`. An actual reviewer must make those attestations; the code does not generate human approval. Citation/schema checks cannot be waived. Changing the target, factsheet, or evidence invalidates the attestation. Normal dataset approval, classification, group, and exclusion checks remain mandatory.

Seed dataset payloads and SQL-application feedback accept `semantic_review`; the feedback API binds the reviewer identity to the authenticated principal. Workbench feedback accepts the same review and carries it in case provenance into preflight. Serving remains conservative even when a training target is admissible.

## Local judge and retrieval qualification

Use `docs/examples/judge-review-template.jsonl` as a format example for the six required challenge categories. Build real, diverse `calibration` and `qualification` partitions, isolated by borrower group from each other, training data, and release benchmarks. No qualification observations may be used for model/prompt/threshold tuning.

Each row requires a nonempty `group_id`, the frozen claim and cited `sources`, and two independently produced human `reviews`, each containing `reviewer_id` and `label` (`supported` or `unsupported`). Disagreement requires an `adjudication` object from a third reviewer. Do not duplicate groups to inflate qualification counts. The tool retains labels and predictions together for audit, but sends only claims and sources to the judge.

```sh
uv run credit-risk-qualify judge --input reviewed-judge-cases.jsonl \
  --candidate-model candidate-model --judge-model separate-local-judge \
  --judge-revision immutable-judge-snapshot-hash --out outputs/judge-qualification.json
```

The judge must use a loopback or `host.docker.internal` oMLX endpoint, a model distinct from the candidate, and a pinned revision. Its rubric, decoding parameters, and identity are frozen in the artifact. Errors, malformed results, fabricated source references, and uncertain judgments never count as support.

Judge qualification requires lower 95% Wilson bounds of at least **0.95** for unsupported-claim detection and **0.90** for supported-claim recognition. Qualification must cover all six challenge categories. Merely agreeing/disagreeing with extractive checks is not a qualification criterion.

For retrieval rows, provide `question`, `passage`, `group_id`, `split`, and the same independently reviewed labels, interpreting `supported` as relevant and `unsupported` as irrelevant. The tool computes actual cosine similarities, selects the threshold using calibration cases only, and assesses qualification cases afterward:

```sh
uv run credit-risk-qualify retrieval --input reviewed-retrieval-cases.jsonl \
  --candidate-model candidate-model --out outputs/retrieval-qualification.json
```

Threshold selection maximizes the worse of relevant-passage recall and irrelevant-passage rejection, breaking ties toward rejection, recall, then the higher threshold. Qualification requires lower Wilson bounds of **0.90** recall and **0.95** rejection. Set `CR_EMBEDDING_REVISION` to the immutable embedding snapshot hash before qualification. Freeze the resulting threshold and embedding signature into the deployed policy and evaluation metadata. A saved `passed` flag is never accepted as proof; checks are recomputed from reviewed observations.

## Release evaluation and comparison

```sh
uv run credit-risk-evaluate --gold frozen-gold.jsonl --model candidate-model \
  --model-revision candidate-checkpoint-sha256 \
  --judge-model separate-local-judge --judge-revision immutable-judge-snapshot-hash \
  --out outputs/candidate-report.json
```

For captured predictions use `--outputs predictions.json --candidate-model candidate-model` in place of `--model`. The prediction file maps case IDs to `{ "output": {...}, "numerics": {...}, "blocked": false, "provider_failed": false }`; `numerics` is an optional projection of **predicted** values, never copied gold answers. Missing predictions become provider failures. Unknown case IDs are rejected. Numeric facts written as `field.path = value` are extracted when no explicit numeric projection is supplied.

Gold records may supply `factsheet`, `factsheet_case_id`, and `group_id`. Without complete group lineage, release qualification stays blocked. The workbench includes the same `release_evaluation` report for held-out credit-analysis cases, in addition to its existing field/consistency report. Query-plan and development cases do not masquerade as a held-out credit release benchmark.

Gate results include nullable observations, eligible independent-group denominators, confidence bounds, and `passed`, `failed`, `not_applicable`, or `insufficient_evidence_to_gate` statuses. Every configured metric needs coverage overall; conditional populations can be inapplicable within individual slices. Groups use their worst observation so related cases do not inflate sample size. Every applicable slice requires at least **30** independent observations. This is a floor, not a promise that 30 cases can establish a high threshold:

- Binary rate gates below 100% use lower 95% Wilson bounds.
- Exact-correctness and zero-error gates require zero observed breaches plus coverage.
- Fractional recall/coverage metrics use a conservative one-sided 95% Hoeffding bound rather than treating fractional scores as independent Bernoulli trials. High thresholds can require substantially more data.

Driver recall is gated at **0.90**; required-evidence recall at **0.98**, semantic support at **0.95**, and answer-status correctness at **1.0**. Empty answers and incorrect abstentions fail usefulness checks. Independently established unsupported claims remain zero-tolerance breaches.

To establish qualification eligibility, supply `--metadata metadata.json` containing:

- `qualification.judge` and `qualification.retrieval`: the full sealed qualification artifacts.
- `qualification.training_groups` and `qualification.calibration_groups`: explicit group inventories.
- `qualification.exclusions`: a sealed object containing those same inventories; use `credit_risk.evaluation.qualification.seal` to freeze the reviewed exclusion manifest.
- `embedding_signature` and `retrieval_threshold`: the actual evaluated retrieval configuration.
- `benchmark_review`: a sealed `{ "reviewer_id": "...", "benchmark_manifest": "..." }` reflecting actual review of the frozen manifest in the first report.
- `generation` when using captured outputs: the original frozen generation settings.

The code verifies hashes, reviewer coverage, disjoint group inventories, and matching judge/retrieval/benchmark identities. These artifacts are local audit records, not cryptographic proof of reviewer identity. Keep them in the governed review workflow. Missing artifacts keep promotion blocked; there is no kill-switch override.

Add `--champion outputs/champion-report.json --target-metric driver_recall` to compare against a sealed v2 champion report. Promotion requires gate success, independent qualification, improvement in the declared target metric, and no material task/portfolio regression (existing tolerance 0.02). It returns a recommendation and never overwrites the champion.

## Verification

```sh
uv lock --check
uv run --frozen ruff check .
uv run --frozen pytest -q
RUN_MLX_TRAINING_SMOKE=1 uv run --frozen pytest -q tests/test_training_native.py
```

CI runs the locked offline suite and pinned Ruff 0.16.6 on Linux. Repository pre-commit hooks run the same lint, lock, and offline checks. The opt-in native test trains a tiny random 4-bit model in temporary storage and checks real MLX mask/checkpoint behavior; it does not measure production-model accuracy. Qdrant's in-process backend tests migration/collision behavior; opt-in live Qdrant/oMLX suites remain separate. Human calibration labels and an expanded release benchmark must be supplied before any qualification or accuracy claim.
