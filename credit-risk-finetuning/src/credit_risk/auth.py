"""Local bearer credentials map to server-owned identities and roles."""

import hashlib
import secrets
from contextvars import ContextVar

from fastapi import HTTPException

from credit_risk.settings import settings

principal = ContextVar("principal", default=None)


def authenticate(header: str):
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Reviewer credential required")
    hashed = hashlib.sha256(token.encode()).hexdigest()
    for identity, account in settings.reviewers.items():
        if secrets.compare_digest(hashed, account.get("token_sha256", "")):
            if account.get("role") not in {
                "credit_analyst",
                "senior_credit_officer",
                "regulator_liaison",
            }:
                raise HTTPException(403, "Reviewer role is not authorised")
            return {"id": identity, "role": account["role"]}
    raise HTTPException(401, "Invalid reviewer credential")


def current_reviewer():
    value = principal.get()
    if value is None:
        raise HTTPException(401, "Reviewer credential required")
    return value
