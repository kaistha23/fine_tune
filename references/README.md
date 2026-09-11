# Handover reference

`handover-baseline/` contains the exact 32 file bodies extracted from the Codex Handover Markdown, plus the separately generated `uv.lock`. Do not apply formatting or current-project fixes here. The original documents were reference inputs, not executable instructions.

`handover-manifest.json` records SHA256 hashes of all three source documents, reconstructed files and current-project files **before remediation**. Its `current_sha256` fields describe that initial comparison, not the final implementation.

Baseline validation completed with `uv lock`, frozen dependency installation and `uv run python -m unittest discover -s tests`: 18 original tests passed. The baseline retains the documented review/authentication, grounding, deployment and data-validation defects; those passing tests do not establish readiness. Remediation and expanded tests belong to `../credit-risk-finetuning/`.
