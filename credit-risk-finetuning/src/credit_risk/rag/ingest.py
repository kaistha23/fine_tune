"""Turn policy and regulatory documents into retrievable chunks.

The retrieval layer consumed PolicyChunk records but nothing produced them, so the index
could only ever be filled by hand.

Chunking is by heading and clause, not by fixed character count. Regulatory text is
addressed by clause - "SAMA circular 4, section 7.2" - and a citation that points at
character offset 4000 is useless to a reviewer who has to verify it. Splitting mid-clause
also strips the condition off its rule, which is how a model ends up asserting an
obligation without its exception.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date

from credit_risk.rag.schemas import ApprovalStatus, ConfidentialityLevel, PolicyChunk
from credit_risk.schemas import Jurisdiction

# A heading is a numbered clause ("7.2 Staging"), or a markdown heading.
_NUMBERED = re.compile(r"^\s{0,3}(?P<num>\d+(?:\.\d+){0,3})\.?\s+(?P<title>\S.*)$")
_MARKDOWN = re.compile(r"^\s{0,3}(?P<hashes>#{1,6})\s+(?P<title>\S.*)$")

# Target sizes from the plans: roughly 400-800 tokens, approximated in characters.
TARGET_CHARS = 2800
MAX_CHARS = 4000
MIN_CHARS = 120


@dataclass
class DocumentMeta:
    """Everything about a document that its chunks inherit."""

    document_id: str
    jurisdiction: Jurisdiction
    document_version: str = "1.0"
    authority: str = ""
    document_type: str = ""
    approval_status: ApprovalStatus = "draft"
    effective_from: date | None = None
    effective_to: date | None = None
    supersedes_document_id: str | None = None
    confidentiality_level: ConfidentialityLevel = "internal"
    allowed_roles: list[str] = field(default_factory=list)
    portfolio: list[str] = field(default_factory=list)
    source_system: str = ""
    source_uri: str = ""


@dataclass
class Section:
    section_id: str
    heading_path: list[str]
    text: str


def _heading(line: str) -> tuple[str, str, int] | None:
    """Return (section_id, title, depth) if the line opens a section."""
    m = _NUMBERED.match(line)
    if m:
        number = m.group("num")
        return number, m.group("title").strip(), number.count(".") + 1
    m = _MARKDOWN.match(line)
    if m:
        return "", m.group("title").strip(), len(m.group("hashes"))
    return None


def split_sections(text: str) -> list[Section]:
    """Split on headings, keeping each clause with its own heading path."""
    sections: list[Section] = []
    path: list[str] = []
    current_id = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(current_id, list(path), body))

    for line in text.splitlines():
        found = _heading(line)
        if found is None:
            buffer.append(line)
            continue
        flush()
        buffer = []
        section_id, title, depth = found
        del path[depth - 1:]
        path.append(f"{section_id} {title}".strip())
        current_id = section_id or ".".join(str(i + 1) for i in range(len(path)))
    flush()
    return sections


def _pack(paragraphs: list[str]) -> list[str]:
    """Group paragraphs up to the target size, splitting only when one is too large."""
    out: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > MAX_CHARS:
            if current:
                out.append(current)
                current = ""
            # A single oversized paragraph is split on sentence ends, not mid-word.
            sentences = re.split(r"(?<=[.;])\s+", paragraph)
            piece = ""
            for sentence in sentences:
                if len(piece) + len(sentence) + 1 > TARGET_CHARS and piece:
                    out.append(piece.strip())
                    piece = ""
                piece = f"{piece} {sentence}".strip()
            if piece:
                out.append(piece.strip())
            continue
        if len(current) + len(paragraph) + 2 > TARGET_CHARS and current:
            out.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        out.append(current)
    return out


def chunk_document(text: str, meta: DocumentMeta) -> list[PolicyChunk]:
    """Chunk one document, carrying full lineage onto every piece."""
    chunks: list[PolicyChunk] = []
    for section in split_sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
        for index, body in enumerate(_pack(paragraphs)):
            if len(body) < MIN_CHARS and len(chunks) and not section.section_id:
                # Fragments too small to stand alone rejoin the previous chunk rather
                # than becoming a citation that says nothing.
                previous = chunks[-1]
                chunks[-1] = previous.model_copy(
                    update={"text": f"{previous.text}\n\n{body}"})
                continue
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            chunks.append(PolicyChunk(
                chunk_id=f"{meta.document_id}:{section.section_id or 'body'}:{index}",
                jurisdiction=meta.jurisdiction,
                document_id=meta.document_id,
                document_version=meta.document_version,
                content_hash=digest,
                authority=meta.authority,
                document_type=meta.document_type,
                approval_status=meta.approval_status,
                effective_from=meta.effective_from,
                effective_to=meta.effective_to,
                supersedes_document_id=meta.supersedes_document_id,
                section_id=section.section_id,
                chunk_index=index,
                heading_path=section.heading_path,
                confidentiality_level=meta.confidentiality_level,
                allowed_roles=list(meta.allowed_roles),
                portfolio=list(meta.portfolio),
                text=body,
                summary=body[:200],
                keywords=[],
                source_system=meta.source_system,
                source_uri=meta.source_uri,
            ))
    return chunks


def deduplicate(chunks: list[PolicyChunk]) -> list[PolicyChunk]:
    """Drop repeats by content hash within a jurisdiction.

    Boilerplate repeated across circulars otherwise crowds out the clause that actually
    answers the question, and inflates apparent evidence coverage.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[PolicyChunk] = []
    for chunk in chunks:
        key = (chunk.jurisdiction.value, chunk.content_hash)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique
