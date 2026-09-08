from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from credit_risk.schemas import Jurisdiction

ConfidentialityLevel = Literal["public", "internal", "confidential", "restricted"]
ApprovalStatus = Literal["draft", "approved", "withdrawn", "superseded"]


class PolicyChunk(BaseModel):
    """A retrievable passage, carrying the lineage the feedback plan requires."""

    chunk_id: str = Field(min_length=1)
    jurisdiction: Jurisdiction
    document_id: str
    document_version: str
    content_hash: str = ""
    authority: str = ""
    document_type: str = ""
    approval_status: ApprovalStatus = "draft"
    effective_from: date | None = None
    effective_to: date | None = None
    supersedes_document_id: str | None = None
    section_id: str = ""
    heading_path: list[str] = Field(default_factory=list)
    page_number: int | None = None
    paragraph_number: int | None = None
    confidentiality_level: ConfidentialityLevel = "internal"
    allowed_roles: list[str] = Field(default_factory=list)
    portfolio: list[str] = Field(default_factory=list)
    text: str
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    source_system: str = ""
    source_uri: str = ""

    @model_validator(mode="after")
    def validate_effective_window(self) -> PolicyChunk:
        if (self.effective_from and self.effective_to
                and self.effective_from > self.effective_to):
            raise ValueError("effective_from must not be after effective_to")
        return self

    @property
    def evidence_id(self) -> str:
        """Stable citation handle, e.g. SAMA-DOC-4#7.2."""
        return f"{self.document_id}#{self.section_id}" if self.section_id else self.document_id


class AccessContext(BaseModel):
    """Who is asking, for what, and as at when.

    Every field here narrows what may be searched. None of them is optional, because a
    default would silently widen access.
    """

    jurisdiction: Jurisdiction
    role: str = Field(min_length=1)
    as_of_date: date
    portfolio: str | None = None
