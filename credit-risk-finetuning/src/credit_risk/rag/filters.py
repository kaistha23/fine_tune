"""Pre-search access filters - the security core of retrieval.

Every constraint here is applied *before* the search runs, so a document the caller may
not see is never scored, never ranked and never read. Filtering after retrieval is not
equivalent: by then the content has already reached the process that will build a prompt.

build_predicate returns a backend-neutral description. Each index translates it into its
own query language, and both translations are tested against the same cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from credit_risk.rag.schemas import AccessContext, PolicyChunk


class AccessPolicyError(PermissionError):
    pass


@dataclass(frozen=True)
class AccessPredicate:
    """What a caller is allowed to search, expressed as data."""

    collection: str
    jurisdiction: str
    approval_status_in: list[str]
    confidentiality_in: list[str]
    effective_on: date
    portfolio: str | None = None
    describe: list[str] = field(default_factory=list)


class RetrievalPolicy:
    def __init__(self, path: str | Path, expected_version: str | None = None):
        with Path(path).open("r", encoding="utf-8") as handle:
            self.data = yaml.safe_load(handle)
        if self.data.get("default_deny") is not True:
            raise AccessPolicyError("Retrieval policy must be default-deny")
        if expected_version and str(self.data.get("version")) != expected_version:
            raise AccessPolicyError(
                f"Retrieval policy version {self.data.get('version')!r} does not match "
                f"required version {expected_version!r}"
            )

    @property
    def version(self) -> str:
        return str(self.data["version"])

    @property
    def settings(self) -> dict[str, Any]:
        return self.data["retrieval"]

    def collection_for(self, jurisdiction: str) -> str:
        """Jurisdiction to physical collection. Unknown jurisdictions have no collection."""
        namespaces = self.data.get("namespaces", {})
        if jurisdiction not in namespaces:
            raise AccessPolicyError(f"No retrieval namespace for jurisdiction: {jurisdiction}")
        return str(namespaces[jurisdiction])

    def levels_for_role(self, role: str) -> list[str]:
        access = self.data.get("role_access", {})
        if role not in access:
            # Default deny: an unknown role sees nothing, rather than everything.
            raise AccessPolicyError(f"Role has no declared retrieval access: {role}")
        return list(access[role])

    def build_predicate(self, context: AccessContext) -> AccessPredicate:
        collection = self.collection_for(context.jurisdiction.value)
        levels = self.levels_for_role(context.role)
        statuses = list(self.data.get("approval_status_allowed", []))
        if not statuses:
            raise AccessPolicyError("Retrieval policy declares no permitted approval status")
        return AccessPredicate(
            collection=collection,
            jurisdiction=context.jurisdiction.value,
            approval_status_in=statuses,
            confidentiality_in=levels,
            effective_on=context.as_of_date,
            portfolio=context.portfolio,
            describe=[
                f"collection={collection}",
                f"jurisdiction={context.jurisdiction.value}",
                f"approval_status in {statuses}",
                f"confidentiality in {levels}",
                f"effective on {context.as_of_date.isoformat()}",
            ] + ([f"portfolio={context.portfolio}"] if context.portfolio else []),
        )


def chunk_is_visible(chunk: PolicyChunk, predicate: AccessPredicate) -> bool:
    """Reference implementation of the predicate, used by the in-memory index.

    The Qdrant translation is checked against this same function in the tests, so the two
    backends cannot drift apart on a security control.
    """
    if chunk.jurisdiction.value != predicate.jurisdiction:
        return False
    if chunk.approval_status not in predicate.approval_status_in:
        return False
    if chunk.confidentiality_level not in predicate.confidentiality_in:
        return False
    if chunk.effective_from and chunk.effective_from > predicate.effective_on:
        return False
    if chunk.effective_to and chunk.effective_to < predicate.effective_on:
        return False
    # Kept as a guard clause rather than inlined, so every access rule reads the same
    # way down the function and a new one is added without restructuring.
    if predicate.portfolio and chunk.portfolio and predicate.portfolio not in chunk.portfolio:  # noqa: SIM103
        return False
    return True
