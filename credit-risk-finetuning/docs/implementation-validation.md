# Local remediation validation — 9 September 2026

The approved foundation has been implemented in the advisory project. The root rating experiment was preserved. The original handover is reconstructed independently under `../references/handover-baseline`; its 32 files and the source documents are hashed in `../references/handover-manifest.json`. Its generated lock, frozen installation and 18 original tests passed. The current project's existing lock was preserved.

## Controls delivered

- Bearer-authenticated analytical and feedback routes; reviewer identity and role come from private server configuration. The local `/review` page holds its credential in memory.
- SQL compilation exclusively in the data service. The reviewer receives parameterised SQL, masked parameters, template hash, request digest, database snapshot hash, schema/policy versions, tables, columns, joins, grain, filters and governed calculations.
- Transactional SQLite review revisions and audit feedback. Approval binds the complete structured plan, parameter values, registry/policy content and database file digest. Execution consumes approval once. Corrections revoke the previous revision and must be validated and approved anew. Invalid corrections remain audited. SQL text cannot be submitted for execution.
- Read-only analytical mounting exclusively in the internal data service, isolated writable audit mounts, explicit credentials and separate runtime/test images.
- Result field/type/range/finite/date/identity/grain checks, query resource limits and governed calculator checks.
- Pre-search document roles, versioned evidence identifiers and complete lexical pagination. Rebuild existing indexes to adopt the new identifiers and access metadata.
- Conservative extractive grounding checks across all narrative fields. Matching a citation ID alone does not establish support. Retrieval cosine scores are candidate similarity, not calibrated evidence quality. Promotion remains disabled.
- Both dataset producers preserve question and evidence. Dataset eligibility requires named review metadata, quality assessment and declared synthetic/masked classification. Group-level assignment, future-group holdout, exact deduplication, frozen-test exclusions and hashed manifests are enforced. Feedback uses the latest revision per interaction.
- Training preflight checks immutable dataset files, cached 4-bit base, token coverage, assistant loss, fresh candidate directories and accumulation-aligned iteration counts. Training records the resolved base, prompt/template hashes, metrics and completion; validation patience protects long runs. Fusing uses the recorded local base and refuses failed runs or existing destinations.

## Native compatibility spike

The 100 synthetic fixtures exercise mechanics only; their deterministic fixture validation is **not SME credit approval**. They cover Retail/SME/Corporate and SAMA/CBUAE labels without claiming regulatory coverage. Group/time isolation produced 60 training, 15 validation and 25 test records.

- Base: cached `mlx-community/Qwen3.5-9B-4bit`, revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`.
- LoRA: rank 16, scale 2, dropout 0.05, learning rate `2e-5`, batch 1, accumulation 8; 24 micro-batches / 3 optimizer updates.
- Sequence limit 2,048; actual maximum **321 tokens**, minimum assistant coverage 75 tokens. This is not a full-context memory benchmark.
- Wall time approximately 176 seconds; peak MLX allocation 10,838,286,034 bytes (10.1 GiB), maximum RSS approximately 11.2 GB. Sampled system swap stayed near 1,479 MiB; command-reported swaps were zero. Other native/Docker workloads remained running.
- Validation loss 1.538 initially and 1.075 at iteration 24. These synthetic losses do not measure credit-risk quality.
- Adapter save/reload and generation passed. Fusion and fused-model reload/generation passed. The candidate has not replaced the configured serving model or champion.

Local artifacts: `outputs/spike-v3/`, `adapters/candidates/spike-v3/`, `models/candidates/spike-v3/`. These large/generated outputs are ignored by Git. Credentials are excluded from Git and Docker contexts.

## Remaining boundaries

**Recommended next phase:** assemble an SME-reviewed benchmark and calibrate retrieval/claim support before any domain training or promotion. Full semantic entailment, calibrated retrieval thresholds, real institution formulas, currency/unit contracts, financial-statement freshness, temporal joins/model-run lineage and trusted upstream group lineage remain unfinished or require approved inputs. Declared masking is not an automated PII certification. Exact deduplication does not detect paraphrase/template-family leakage. Hash-based 70/15/15 allocation is approximate and group/time holdouts can change proportions.

The legacy gold fixture alone is not a complete reviewed benchmark. Missing coverage, malformed outputs and empty evaluations fail closed; model comparisons require identical case sets and benchmark manifests. No candidate can be promoted by current gates. Free-form supported paraphrases may be rejected by the deliberately conservative extractive validator.

SQL approval is single-use and process-local policy changes require service restart. Snapshot hashing detects ordinary file replacement; the local operating contract requires an immutable analytical file during review/execution. This is a single-reviewer development environment, not a multi-tenant production authorization system.

## Verification results

- Host offline suite: **279 passed, 34 skipped**, plus nine subtests.
- Actual test image: **279 passed, 34 skipped**, plus nine subtests.
- Live Qdrant parity, ACL and pagination suite: **14 passed**. These tests account for 14 of the offline skips; other optional integrations remain unverified.
- Running Compose smoke: review page, preparation, approval and execution passed. Changed borrower, replay, executable SQL and missing authentication were blocked. Valid corrections required a new approval and then executed successfully.
- Additional regressions prove schema, policy and snapshot drift invalidate approval, and validated corrections are retained as typed audit feedback.
- Ruff checks for changed Python files, `git diff --check`, current offline lock validation and all 32 baseline hashes passed. Dependency deprecation warnings remain; no test failures were suppressed.

API and data-service are healthy and Qdrant remains running. The local review page is available at `http://127.0.0.1:8080/review`; obtain its credential from the private `.reviewer-token` file. Source changes remain uncommitted for review.
