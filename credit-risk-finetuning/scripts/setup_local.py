"""Create private local credentials once; never overwrite an existing configuration."""

import hashlib
import json
import os
import secrets
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = [root / ".env", root / ".reviewer-token"]
if any(path.exists() for path in paths):
    raise SystemExit("Existing credentials preserved; configure missing values manually.")
token = secrets.token_urlsafe(32)
reviewers = {
    "local-reviewer": {
        "role": "credit_analyst",
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
    }
}
content = (
    "CR_SERVICE_TOKEN="
    + secrets.token_urlsafe(32)
    + "\nCR_REVIEWERS="
    + json.dumps(reviewers)
    + "\nCR_EMBEDDING_MODEL=\n"
    + "CR_EMBEDDING_REVISION=\n"
    + "CR_OMLX_API_KEY=\n"
)
for path, value in zip(paths, [content, token + "\n"]):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        handle.write(value)
for directory in ("data/audit/api", "data/audit/reviews", "data/training"):
    (root / directory).mkdir(parents=True, exist_ok=True)
print("Created private local credentials. Use .reviewer-token in the local review page.")
