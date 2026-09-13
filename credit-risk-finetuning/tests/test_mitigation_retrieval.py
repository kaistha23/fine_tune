from datetime import date

import pytest

from credit_risk.rag.filters import RetrievalPolicy
from credit_risk.rag.index import InMemoryPolicyIndex
from credit_risk.rag.ingest import DocumentMeta, chunk_document
from credit_risk.rag.retriever import PolicyRetriever, RetrievalError
from credit_risk.rag.schemas import AccessContext, PolicyChunk
from credit_risk.settings import Settings


def chunk(n):
    return PolicyChunk(chunk_id=str(n), section_key=str(n), document_id="d", document_version="1",
                       jurisdiction="SAMA", approval_status="approved", text="Risk guidance " + str(n))


def test_sibling_and_repeated_headings_are_unique_and_stable():
    text = "# Policy\n## Scope\nFirst clause.\n## Limits\nSecond clause.\n## Scope\nThird clause."
    meta = DocumentMeta("D", jurisdiction="SAMA", approval_status="approved")
    # Real ingestion uses the enum, as the reindex loader does.
    from credit_risk.schemas import Jurisdiction
    meta.jurisdiction = Jurisdiction.SAMA
    chunks = chunk_document(text, meta)
    assert len(chunks) == len({c.chunk_id for c in chunks}) == len({c.evidence_id for c in chunks}) == 3
    assert chunks == chunk_document(text, meta)
    index = InMemoryPolicyIndex()
    index.upsert(chunks, collection="policy_sama")
    index.upsert(chunks, collection="policy_sama")
    assert len(index._collections["policy_sama"]) == 3
    with pytest.raises(ValueError, match="Conflicting"):
        index.upsert([chunks[0].model_copy(update={"text": "Changed"})], collection="policy_sama")


def test_filter_before_truncate_and_either_arm():
    class Index:
        def search_dense(self, *args):
            return [(chunk(i), .9) for i in range(9)]
        def search_lexical(self, *args):
            return [(chunk(9), 2.)]
        def score_candidates(self, query, chunks):
            return {c.chunk_id: .9 if int(c.chunk_id) >= 8 else .2 for c in chunks}
    policy = RetrievalPolicy("configs/retrieval.yaml")
    policy.settings["top_k"] = 2
    context = AccessContext(jurisdiction="SAMA", role="credit_analyst", as_of_date=date(2025, 1, 1))
    output = PolicyRetriever(policy, Index()).retrieve("risk", context)
    assert {e.section for e in output["evidence"]} == {""}
    assert len(output["evidence"]) == 2
    assert {e.text for e in output["evidence"]} == {"Risk guidance 8", "Risk guidance 9"}
    assert all(e.score == .9 for e in output["evidence"])


def test_leaked_access_candidate_fails_before_scoring():
    class Index:
        def search_dense(self, *args):
            return [(chunk(1).model_copy(update={"approval_status": "draft"}), 1)]
        def search_lexical(self, *args):
            return []
        def score_candidates(self, *args):
            raise AssertionError("Unauthorized candidate must not be embedded")
    with pytest.raises(RetrievalError):
        PolicyRetriever(RetrievalPolicy("configs/retrieval.yaml"), Index()).retrieve(
            "x", AccessContext(jurisdiction="SAMA", role="credit_analyst", as_of_date=date(2025, 1, 1)))


def test_hashing_requires_explicit_test_mode():
    from credit_risk.rag.factory import build_embedder
    with pytest.raises(ValueError, match="CR_EMBEDDING_MODEL"):
        build_embedder(Settings(_env_file=None, offline_test_mode=False, embedding_model=""))
    with pytest.raises(ValueError, match="CR_EMBEDDING_REVISION"):
        build_embedder(Settings(_env_file=None, offline_test_mode=False,
                                embedding_model="semantic-model", embedding_revision=""))
    assert build_embedder(Settings(_env_file=None, offline_test_mode=True)).signature.startswith("placeholder")


def test_unknown_environment_setting_rejected(monkeypatch):
    monkeypatch.setenv("CR_EMBEDING_MODEL", "typo")
    with pytest.raises(ValueError, match="CR_EMBEDING_MODEL"):
        Settings(_env_file=None)


def test_qdrant_backend_reindex_and_collision_checks():
    qdrant = pytest.importorskip("qdrant_client")
    from credit_risk.rag.embedding import HashingEmbedder
    from credit_risk.rag.index import QdrantPolicyIndex
    from credit_risk.rag.reindex import rebuild
    index = QdrantPolicyIndex.__new__(QdrantPolicyIndex)
    index.client = qdrant.QdrantClient(":memory:")
    index.embedder = HashingEmbedder()
    documents = [{"text": "# Policy\n## One\nFirst passage.\n## Two\nSecond passage.",
                  "meta": {"document_id": "D", "jurisdiction": "SAMA", "approval_status": "approved"}}]
    namespaces = {"SAMA": "test_v2", "CBUAE": "test_cbuae_v2"}
    report = rebuild(documents, index, namespaces)
    assert report["chunk_count"] == 2
    with pytest.raises(ValueError, match="new collections"):
        rebuild(documents, index, namespaces)
    c = chunk(42)
    index.upsert([c], collection="test_v2")
    index.upsert([c], collection="test_v2")
    with pytest.raises(ValueError, match="Conflicting"):
        index.upsert([c.model_copy(update={"text": "changed"})], collection="test_v2")
    index.client.close()


def test_skipped_heading_levels_preserve_sibling_paths():
    from credit_risk.rag.ingest import split_sections
    sections = split_sections("# Root\n### A\nFirst\n### B\nSecond")
    assert sections[0].heading_path == ["Root", "A"]
    assert sections[1].heading_path == ["Root", "B"]


def test_embedding_signature_records_revision_and_query_instruction():
    from credit_risk.rag.embedding import OMLXEmbedder
    a = OMLXEmbedder("http://localhost/v1", "embedding", model_revision="rev-a")
    b = OMLXEmbedder("http://localhost/v1", "embedding", model_revision="rev-b")
    a._dimensions = b._dimensions = 3
    assert a.signature != b.signature
    b.model_revision = "rev-a"
    b.query_instruction = "Different retrieval task"
    assert a.signature != b.signature


def test_health_returns_unavailable_when_embedding_backend_fails(monkeypatch):
    from credit_risk import api
    from credit_risk.settings import settings
    from fastapi import HTTPException
    monkeypatch.setattr(settings, "offline_test_mode", False)
    class Unavailable:
        def embed_query(self, text):
            raise RuntimeError("service unavailable")
    monkeypatch.setattr(api.retriever.index, "embedder", Unavailable())
    with pytest.raises(HTTPException) as error:
        api.health()
    assert error.value.status_code == 503
