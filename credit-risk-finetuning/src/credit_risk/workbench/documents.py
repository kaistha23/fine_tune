"""Versioned policy and regulation documents for workbench retrieval.

Files are stored immutably by sha256 and registered as ``document_id@version`` records that
are never edited. Supersession is derived, not stored: when a newer approved version of the
same document (or a document declaring ``supersedes_document_id``) becomes effective, the
older version's effective window ends the day before. Retrieval then applies the existing
effective-date filter, so a question as at an earlier date still sees the older text.

Extraction and chunking run on registration (CPU). Embedding runs later in the single model
lane and is cached per chunk and embedder signature, so rebuilding the index never re-embeds.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from credit_risk.rag.extract import ExtractionError, extract
from credit_risk.rag.filters import RetrievalPolicy
from credit_risk.rag.index import InMemoryPolicyIndex
from credit_risk.rag.ingest import DocumentMeta, chunk_pages, deduplicate
from credit_risk.rag.retriever import PolicyRetriever
from credit_risk.rag.schemas import ApprovalStatus, ConfidentialityLevel, PolicyChunk
from credit_risk.schemas import Jurisdiction

SUFFIXES = {".pdf", ".docx", ".md", ".txt"}
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"


class DocumentError(ValueError):
    pass


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(pattern=IDENTIFIER)
    version: str = Field(pattern=IDENTIFIER)
    jurisdiction: Jurisdiction
    title: str = Field(default="", max_length=300)
    document_type: str = Field(default="policy", max_length=64)
    authority: str = Field(default="", max_length=128)
    approval_status: ApprovalStatus = "approved"
    effective_from: date
    effective_to: date | None = None
    supersedes_document_id: str | None = Field(default=None, pattern=IDENTIFIER)
    confidentiality_level: ConfidentialityLevel = "internal"
    allowed_roles: list[str] = Field(default_factory=list, max_length=10)
    portfolio: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("portfolio")
    @classmethod
    def known_portfolios(cls, value):
        if set(value) - {"retail", "sme", "corporate"}:
            raise ValueError("Portfolio must be retail, sme or corporate")
        return sorted(set(value))

    @field_validator("effective_to")
    @classmethod
    def ordered_window(cls, value, info):
        start = info.data.get("effective_from")
        if value and start and value < start:
            raise ValueError("effective_to must not be before effective_from")
        return value


class CachedEmbedder:
    """Reuse stored document vectors; only queries (and new chunks) reach the model."""

    def __init__(self, inner, vectors: dict[str, list[float]]):
        self.inner = inner
        self.vectors = vectors

    @property
    def dimensions(self):
        return self.inner.dimensions

    @property
    def signature(self):
        return self.inner.signature

    def embed(self, text):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self.vectors:
            self.vectors[key] = self.inner.embed(text)
        return self.vectors[key]

    def embed_query(self, text):
        return self.inner.embed_query(text)


class DocumentLibrary:
    def __init__(self, store, root: Path, retrieval_policy: Path):
        self.store = store
        self.root = Path(root)
        self.files = self.root / "files"
        self.staging = self.root / "staging"
        self.index_root = self.root / "index"
        self.retrieval_policy = Path(retrieval_policy)

    # -- upload --------------------------------------------------------------------------
    def stage(self, content: bytes, filename: str) -> dict:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUFFIXES:
            raise DocumentError("Upload PDF, DOCX, Markdown or plain text")
        if not content:
            raise DocumentError("Uploaded document is empty")
        if len(content) > MAX_DOCUMENT_BYTES:
            raise DocumentError("Uploaded document exceeds 50 MB")
        digest = hashlib.sha256(content).hexdigest()
        self.staging.mkdir(parents=True, exist_ok=True)
        target = self.staging / (digest + suffix)
        if not target.exists():
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
        return {"staged_id": target.name, "file_name": Path(filename).name, "file_sha256": digest}

    def _staged(self, staged_id: str) -> Path:
        path = (self.staging / staged_id).resolve()
        if path.parent != self.staging.resolve() or not path.is_file():
            raise DocumentError("Unknown staged document")
        return path

    def register(self, staged_id: str, file_name: str, metadata: DocumentMetadata) -> dict:
        staged = self._staged(staged_id)
        digest = staged.name.split(".")[0]
        key = f"{metadata.document_id}@{metadata.version}"
        if any(
            item["id"] == key and item["file_sha256"] != digest for item in self.store.list("document")
        ):
            raise DocumentError(f"{key} is already registered with different content; use a new version")
        if metadata.approval_status == "approved" and any(
            item["id"] != key
            and item["approval_status"] == "approved"
            and item["effective_from"] == metadata.effective_from.isoformat()
            and (
                item["document_id"] == metadata.document_id
                or item["document_id"] == metadata.supersedes_document_id
                or item.get("supersedes_document_id") == metadata.document_id
            )
            for item in self.store.list("document")
        ):
            raise DocumentError(
                "Another approved version takes effect on the same date; choose a later effective_from"
            )
        try:
            pages = extract(staged)
        except (ExtractionError, OSError, ValueError) as exc:
            raise DocumentError("Document text could not be extracted: " + str(exc)) from exc
        chunks = deduplicate(chunk_pages(pages, self._meta(metadata, file_name)))
        if not chunks:
            raise DocumentError("Document produced no retrievable text")
        self.files.mkdir(parents=True, exist_ok=True)
        stored = self.files / staged.name
        if not stored.exists():
            staged.replace(stored)
        record = self.store.add(
            "document",
            {
                **metadata.model_dump(mode="json"),
                "file_name": Path(file_name).name,
                "file_sha256": digest,
                "stored_file": stored.name,
                "pages": len(pages),
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            },
            key,
        )
        return self.describe(record)

    def _meta(self, metadata: DocumentMetadata, file_name: str) -> DocumentMeta:
        return DocumentMeta(
            document_id=metadata.document_id,
            jurisdiction=metadata.jurisdiction,
            document_version=metadata.version,
            authority=metadata.authority,
            document_type=metadata.document_type,
            approval_status=metadata.approval_status,
            effective_from=metadata.effective_from,
            effective_to=metadata.effective_to,
            supersedes_document_id=metadata.supersedes_document_id,
            confidentiality_level=metadata.confidentiality_level,
            allowed_roles=list(metadata.allowed_roles),
            portfolio=list(metadata.portfolio),
            source_system="workbench-upload",
            source_uri=f"workbench://documents/{metadata.document_id}@{metadata.version}/{Path(file_name).name}",
        )

    # -- lifecycle -----------------------------------------------------------------------
    def effective_windows(self, documents: list[dict] | None = None) -> dict[str, dict]:
        """Declared windows narrowed by later approved versions and superseding documents."""
        documents = documents if documents is not None else self.store.list("document")
        approved = [item for item in documents if item["approval_status"] == "approved"]
        windows = {}
        for item in documents:
            start = date.fromisoformat(item["effective_from"])
            end = date.fromisoformat(item["effective_to"]) if item["effective_to"] else None
            superseded_by = None
            for other in approved:
                if other["id"] == item["id"]:
                    continue
                same_document = other["document_id"] == item["document_id"]
                replaces = other.get("supersedes_document_id") == item["document_id"]
                other_start = date.fromisoformat(other["effective_from"])
                if (same_document or replaces) and other_start > start:
                    candidate = other_start - timedelta(days=1)
                    if end is None or candidate < end:
                        end, superseded_by = candidate, other["id"]
            windows[item["id"]] = {
                "effective_from": start.isoformat(),
                "effective_to": end.isoformat() if end else None,
                "superseded_by": superseded_by,
            }
        return windows

    def describe(self, record: dict, windows: dict | None = None, signature: str | None = None) -> dict:
        windows = windows or self.effective_windows()
        summary = {k: v for k, v in record.items() if k != "chunks"}
        summary["chunk_count"] = len(record["chunks"])
        summary["effective_window"] = windows.get(record["id"])
        summary["indexed_with"] = sorted(self._indexed_signatures(record))
        if signature:
            summary["indexed"] = signature in summary["indexed_with"]
        return summary

    def list(self) -> list[dict]:
        documents = self.store.list("document")
        windows = self.effective_windows(documents)
        return [self.describe(item, windows) for item in documents]

    # -- embedding cache -----------------------------------------------------------------
    def _index_file(self, record: dict, signature: str) -> Path:
        tag = hashlib.sha256(signature.encode()).hexdigest()[:16]
        return self.index_root / f"{record['document_id']}@{record['version']}--{tag}.json"

    def _indexed_signatures(self, record: dict) -> set[str]:
        found = set()
        if not self.index_root.is_dir():
            return found
        for path in self.index_root.glob(f"{record['document_id']}@{record['version']}--*.json"):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("file_sha256") == record["file_sha256"]:
                found.add(payload["signature"])
        return found

    def pending(self, signature: str) -> list[dict]:
        return [item for item in self.store.list("document") if signature not in self._indexed_signatures(item)]

    def index(self, embedder, progress=None) -> list[str]:
        """Embed every document not yet embedded with this embedder."""
        pending = self.pending(embedder.signature)
        self.index_root.mkdir(parents=True, exist_ok=True)
        done = []
        for number, record in enumerate(pending, start=1):
            vectors = {}
            for chunk in record["chunks"]:
                key = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
                if key not in vectors:
                    vectors[key] = embedder.embed(chunk["text"])
            target = self._index_file(record, embedder.signature)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "document": record["id"],
                        "file_sha256": record["file_sha256"],
                        "signature": embedder.signature,
                        "vectors": vectors,
                    }
                )
            )
            temporary.replace(target)
            done.append(record["id"])
            if progress:
                progress(number, len(pending))
        return done

    # -- retrieval -----------------------------------------------------------------------
    def retriever(self, embedder, require_indexed: bool = True) -> tuple[PolicyRetriever, dict]:
        """An in-memory hybrid retriever over every registered version, windows derived."""
        policy = RetrievalPolicy(self.retrieval_policy)
        documents = self.store.list("document")
        windows = self.effective_windows(documents)
        vectors: dict[str, list[float]] = {}
        chunks: list[PolicyChunk] = []
        versions = {}
        for record in documents:
            index_file = self._index_file(record, embedder.signature)
            if index_file.is_file():
                payload = json.loads(index_file.read_text())
                if payload.get("file_sha256") == record["file_sha256"]:
                    vectors.update(payload["vectors"])
            elif require_indexed:
                raise DocumentError(
                    f"{record['id']} is not indexed with {embedder.signature}; run Index documents"
                )
            window = windows[record["id"]]
            for chunk in record["chunks"]:
                chunks.append(
                    PolicyChunk.model_validate(
                        {**chunk, "effective_to": window["effective_to"]}
                    )
                )
            versions[record["id"]] = {"file_sha256": record["file_sha256"], **window}
        index = InMemoryPolicyIndex(CachedEmbedder(embedder, vectors))
        if chunks:
            index.upsert(chunks, namespaces=policy.data["namespaces"])
        return PolicyRetriever(policy, index), versions
