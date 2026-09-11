# Workbench setup validation

Implemented for the user's pipeline-only phase. Phase-2 data creation and substantive training/evaluation were not performed. The pre-existing adapter spike and SQL application were preserved.

## Delivered

- Native localhost dashboard on port 8090: Datasets, Runs, Evaluation, Answers & feedback. No manual login or reviewer token.
- Explicit dataset registration, checksum/group/template/OOT controls, tokenizer preflight and separate task adapters.
- Durable native job queue with one model worker lane, explicit start/stop, interrupted-run recovery, frozen run specs and checkpoint lineage.
- Training/validation loss charts and run telemetry; separate validation/test/OOT scorecards; compatible base/adapter and separately labeled prompt comparisons.
- Field-based numerical/driver/citation/abstention checks, query-plan/compiler/result metrics and real-provider repeated/paraphrase/negative-control harnesses.
- Submit-only feedback, including comment-only reports and invalid corrections retained with diagnostics. No approve/accept/reject stages. Latest valid model corrections can enter a future batch; frozen groups cannot.
- Feedback development checks, immutable prompt/schema versions, explicit version selection/rollback and unapplied checklist recommendations.
- Source SQL inspection and parameter masking; only structured plans can reach deterministic synthetic-query execution.
- Training/inference output-schema alignment; final-checkpoint validation, best-checkpoint selection and hashed export lineage.
- Explicit Qwen non-thinking mode across tokenizer preflight, MLX training, evaluation and oMLX serving requests.
- Bounded LoRA/optimizer/schedule controls, recorded 4-bit affine quantization metadata, non-finite-loss failure and early-stopping margin/patience.
- V2 dataset lineage, typed facts, independent held-out expectations and versioned privacy-scan evidence for masked data.
- Bootstrap 95% metric intervals, minimum-sample labels, paired candidate comparisons, abstention precision, confidence Brier scoring and business-severity weighting.
- Fail-closed feedback eligibility: unknown causes remain untriaged and a correction cannot certify its own expected result.

## Evidence

Host offline suite: **323 passed, 34 skipped, plus nine subtests**. The workbench suite contributes 42 tests. Checks include invalid-output metric denominators, confidence intervals and calibration, unit/date tolerances, V2 lineage/privacy controls, unknown-feedback exclusion, feedback deduplication, duplicate submissions, latest-correction selection, protected feedback, schema compatibility and non-activating migrations, cross-origin requests, repeated/paraphrased questions, changed-question false matches, job queue/recovery/stop, checkpoint-export selection and a real one-row temporary DuckDB result comparison.

The browser smoke passed all four views, empty states, answer import, direct feedback, masked SQL, version editing without implicit activation, independent validation/test/OOT scorecards, loss-chart rendering and mobile width. No JavaScript errors occurred. Browser fixtures lived in a temporary workspace, with no model load or training job. Screenshots remain under `outputs/workbench-browser/` and contain only test fixtures.

A native tokenizer-only preflight using two temporary fixtures passed: maximum 636 tokens and minimum 23 assistant tokens. Preflight matched the installed MLX ChatDataset token sequence and loss-mask boundary. It loaded no model weights and started no training. This is not a full-context memory benchmark.

Ruff, JavaScript syntax, current offline lock validation and whitespace checks pass. `jsonschema` was added as a pinned dependency for versioned output validation. Framework/dependency deprecation warnings remain visible in test logs.

## Practical limits

- The live workspace contains no phase-2 datasets, jobs, answers or feedback. Accuracy is **Not evaluated** until datasets and expected checks are supplied.
- Unit tests with controlled providers verify consistency detection, not the consistency of a trained model. The real-model harness will run three generations per question after an explicit evaluation job is started.
- Credit support checking remains a conservative extractive heuristic; it can reject supported paraphrases and does not measure expert judgment. Driver/field agreement needs canonical phase-2 labels.
- Recommendations are failure-category-specific hypotheses, not automatic root-cause proof or measured prompt improvements. They never change the active version on submission.
- Existing core response fields are preserved. Breaking output-schema changes are saved for inspection but cannot activate without a contract migration; compatible additions still need valid targets/checks.
- Masking, group identity and template-family declarations need correct upstream metadata. Exact deduplication is not a semantic leakage detector.
- Feedback-derived regression cases are development checks. They cannot establish independent held-out accuracy, even when they pass.
- No training, adapter fusion, serving-model replacement or model-weight evaluation was launched in this setup phase.

## Repeat checks

```sh
uv run --no-sync pytest -q
uv run --no-sync pytest -q tests/workbench
node --check src/credit_risk/workbench/static/app.js
uv tool run --from playwright playwright install chromium
uv tool run --from playwright python scripts/check_workbench_browser.py
uv lock --check --offline
```

The workbench is started with `uv run --no-sync python -m credit_risk.workbench.server`. See [the phase-2 data contract](workbench-data-contract.md) before registering data. Source changes remain uncommitted for review.

Supplemental Linux container check: **309 passed, 35 skipped, plus nine subtests**, using the built test image with final `src/` and `tests/` mounted read-only. The extra skip compared with the host is the cached Mac model preflight test. An earlier build captured source before the invalid-output denominator fix; the final-source rerun passed. This container check does not imply Linux MLX training support.

Final live check: `http://127.0.0.1:8090` responds with the expanded tuning and generation-profile controls. Browser test fixtures were isolated and removed. A real-model consistency measurement remains Not evaluated until phase-2 data is provided and an evaluation job is explicitly started.
