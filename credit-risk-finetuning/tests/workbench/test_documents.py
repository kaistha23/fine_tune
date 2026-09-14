import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from credit_risk.rag.embedding import HashingEmbedder
from credit_risk.rag.schemas import AccessContext
from credit_risk.workbench.documents import DocumentError, DocumentLibrary, DocumentMetadata
from credit_risk.workbench.store import Store

ROOT = Path(__file__).resolve().parents[2]
MONITORING_V1 = "# Watchlist\n\n7.2 Enhanced monitoring\n\nStage 2 obligors require enhanced monitoring reviewed every quarter by the credit committee.\n"
MONITORING_V2 = "# Watchlist\n\n7.2 Enhanced monitoring\n\nStage 2 obligors require enhanced monitoring reviewed every month by the credit committee.\n"


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.document_calls = 0

    def embed(self, text):
        self.document_calls += 1
        return super().embed(text)


@pytest.fixture
def library(tmp_path):
    policy = yaml.safe_load((ROOT / "configs/retrieval.yaml").read_text())
    # Hashing vectors are not semantic; the relevance gate is exercised elsewhere.
    policy["retrieval"]["min_score"] = 0.0
    path = tmp_path / "retrieval.yaml"
    path.write_text(yaml.safe_dump(policy))
    return DocumentLibrary(Store(tmp_path / "workspace"), tmp_path / "documents", path)


def add(library, text, **fields):
    metadata = DocumentMetadata(**{"jurisdiction": "SAMA", "version": "1.0", **fields})
    staged = library.stage(text.encode(), "policy.md")
    return library.register(staged["staged_id"], "policy.md", metadata)


def retrieve(library, embedder, as_of, jurisdiction="SAMA", question="stage 2 enhanced monitoring"):
    retriever, versions = library.retriever(embedder)
    result = retriever.retrieve(
        question, AccessContext(jurisdiction=jurisdiction, role="credit_analyst", as_of_date=as_of)
    )
    return {(item.document_id, item.document_version) for item in result["evidence"]}, versions


def test_new_version_supersedes_on_its_effective_date(library):
    add(library, MONITORING_V1, document_id="SAMA-WATCH", effective_from=date(2025, 1, 1))
    add(library, MONITORING_V2, document_id="SAMA-WATCH", version="2.0", effective_from=date(2026, 1, 1))
    embedder = HashingEmbedder()
    library.index(embedder)
    earlier, versions = retrieve(library, embedder, date(2025, 6, 30))
    later, _ = retrieve(library, embedder, date(2026, 2, 1))
    assert earlier == {("SAMA-WATCH", "1.0")}
    assert later == {("SAMA-WATCH", "2.0")}
    assert versions["SAMA-WATCH@1.0"]["effective_to"] == "2025-12-31"
    assert versions["SAMA-WATCH@1.0"]["superseded_by"] == "SAMA-WATCH@2.0"
    assert versions["SAMA-WATCH@2.0"]["effective_to"] is None


def test_superseding_document_and_access_rules(library):
    add(library, MONITORING_V1, document_id="SAMA-OLD", effective_from=date(2025, 1, 1))
    add(
        library,
        MONITORING_V2,
        document_id="SAMA-NEW",
        supersedes_document_id="SAMA-OLD",
        effective_from=date(2025, 7, 1),
    )
    add(library, MONITORING_V2 + "\nDraft wording.", document_id="SAMA-DRAFT", approval_status="draft", effective_from=date(2025, 1, 1))
    add(library, MONITORING_V1, document_id="CBUAE-WATCH", jurisdiction="CBUAE", effective_from=date(2025, 1, 1))
    add(
        library,
        MONITORING_V1 + "\nBoard only.",
        document_id="SAMA-SECRET",
        confidentiality_level="restricted",
        effective_from=date(2025, 1, 1),
    )
    embedder = HashingEmbedder()
    library.index(embedder)
    assert retrieve(library, embedder, date(2025, 3, 1))[0] == {("SAMA-OLD", "1.0")}
    assert retrieve(library, embedder, date(2025, 8, 1))[0] == {("SAMA-NEW", "1.0")}
    assert retrieve(library, embedder, date(2025, 8, 1), "CBUAE")[0] == {("CBUAE-WATCH", "1.0")}


def test_index_is_required_cached_and_documents_are_immutable(library):
    add(library, MONITORING_V1, document_id="SAMA-WATCH", effective_from=date(2025, 1, 1))
    embedder = CountingEmbedder()
    with pytest.raises(DocumentError, match="not indexed"):
        library.retriever(embedder)
    assert library.index(embedder) == ["SAMA-WATCH@1.0"]
    assert library.index(embedder) == []
    embedded = embedder.document_calls
    library.retriever(embedder)
    assert embedder.document_calls == embedded
    assert library.list()[0]["indexed_with"] == [embedder.signature]
    with pytest.raises(DocumentError, match="different content"):
        add(library, MONITORING_V2, document_id="SAMA-WATCH", effective_from=date(2025, 1, 1))
    assert add(library, MONITORING_V1, document_id="SAMA-WATCH", effective_from=date(2025, 1, 1))["id"] == "SAMA-WATCH@1.0"
    with pytest.raises(DocumentError, match="same date"):
        add(library, MONITORING_V2, document_id="SAMA-WATCH", version="1.1", effective_from=date(2025, 1, 1))
    with pytest.raises(DocumentError, match="PDF, DOCX"):
        library.stage(b"x", "policy.exe")
    with pytest.raises(DocumentError, match="Unknown staged"):
        library.register("../workbench.sqlite3", "x.md", DocumentMetadata(document_id="X", version="1", jurisdiction="SAMA", effective_from=date(2025, 1, 1)))
    with pytest.raises(ValueError):
        DocumentMetadata(document_id="../escape", version="1", jurisdiction="SAMA", effective_from=date(2025, 1, 1))


def test_document_endpoints_queue_index_job_and_worker_indexes(tmp_path, monkeypatch):
    from credit_risk.workbench import server, worker

    model = tmp_path / "embedder"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"hidden_size": 256}))
    catalog = [{"id": "fixture", "path": str(model), "label": "Fixture embedder"}]
    monkeypatch.setattr(server, "embedding_catalog", lambda: catalog)
    app = server.create_app(tmp_path / "workspace", False)
    client = TestClient(app)
    client.headers["X-Workbench-Token"] = client.get("/api/session").json()["token"]
    staged = client.post("/api/documents/stage?filename=watch.md", content=MONITORING_V1.encode()).json()
    metadata = {"document_id": "SAMA-WATCH", "version": "1.0", "jurisdiction": "SAMA", "effective_from": "2025-01-01"}
    registered = client.post("/api/documents", json={"staged_id": staged["staged_id"], "file_name": "watch.md", "metadata": metadata})
    assert registered.status_code == 200, registered.text
    listed = client.get("/api/documents").json()
    assert listed["documents"][0]["chunk_count"] >= 1 and listed["documents"][0]["indexed"] is False
    job = client.post("/api/documents/index", json={}).json()
    assert job["spec"]["kind"] == "index_documents" and job["spec"]["documents"] == ["SAMA-WATCH@1.0"]
    assert client.post("/api/documents/index", json={}).status_code == 422

    class FixtureEmbedder(HashingEmbedder):
        def __init__(self, path):
            super().__init__()
            self.path = path

        @property
        def signature(self):
            return f"{self.path}:{self.dimensions}"

    monkeypatch.setattr("credit_risk.rag.embedding.MLXEmbedder", FixtureEmbedder)
    worker.run(job["spec"])
    assert json.loads((Path(job["spec"]["output"]) / "result.json").read_text())["indexed"] == ["SAMA-WATCH@1.0"]
    assert client.get("/api/documents").json()["documents"][0]["indexed"] is True
    detail = client.get("/api/jobs/" + job["id"]).json()
    assert detail["progress"]["total_cases"] == 1 and detail["timing"]["unit"] == "document"
