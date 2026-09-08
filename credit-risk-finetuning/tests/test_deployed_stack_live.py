"""The stack a container actually builds: settings -> oMLX embeddings -> Qdrant -> evidence.

Every other live suite exercises one component. This one exercises the wiring, which is
where the real defect was: compose set CR_QDRANT_URL, Settings had no field for it, and
api.py used PolicyRetriever's default - an empty in-memory index. Each piece worked and
the assembled system retrieved nothing.

Needs both a Qdrant and an oMLX server, so it skips unless both are named:

    docker run --rm -d -p 6399:6333 qdrant/qdrant:v1.19.1
    omlx serve --model-dir <dir with an embedding model> --port 9906 --api-key <key>

    CR_TEST_QDRANT_URL=http://127.0.0.1:6399 \\
    CR_TEST_OMLX_EMBEDDER=mlx-community--Qwen3-Embedding-0.6B-8bit \\
    CR_TEST_OMLX_URL=http://127.0.0.1:9906/v1 \\
    CR_OMLX_API_KEY=<key> \\
        uv run python -m pytest tests/test_deployed_stack_live.py -q
"""
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

from credit_risk.rag import factory
from credit_risk.rag.embedding import OMLXEmbedder
from credit_risk.rag.extract import extract
from credit_risk.rag.index import QdrantPolicyIndex
from credit_risk.rag.ingest import DocumentMeta, chunk_pages, deduplicate
from credit_risk.rag.schemas import AccessContext
from credit_risk.schemas import Jurisdiction
from credit_risk.settings import Settings

QDRANT = os.environ.get("CR_TEST_QDRANT_URL")
EMBEDDER = os.environ.get("CR_TEST_OMLX_EMBEDDER")
OMLX = os.environ.get("CR_TEST_OMLX_URL", "http://127.0.0.1:9905/v1")
API_KEY = os.environ.get("CR_OMLX_API_KEY", "")

NAMESPACES = {"SAMA": "test_deployed_sama", "CBUAE": "test_deployed_cbuae"}

SAMA_CIRCULAR = """1. Scope

This circular applies to all licensed banks operating in the Kingdom.

7.2 Significant increase in credit risk

An exposure must be reclassified to stage 2 when its creditworthiness has deteriorated
materially since initial recognition.

9.1 Collateral

Eligible financial collateral is revalued at least quarterly and the haircut applied
follows the supervisory schedule.
"""

CBUAE_CIRCULAR = """4.1 Staging

Banks licensed in the UAE shall reclassify exposures to stage 2 on a significant
increase in credit risk since origination.
"""

PARAPHRASE = "At what point must a loan be downgraded for deteriorating creditworthiness?"


def settings() -> Settings:
    return Settings(
        qdrant_url=QDRANT,
        embedding_model=EMBEDDER,
        omlx_base_url=OMLX,
        omlx_api_key=API_KEY,
    )


def ingest(text: str, jurisdiction: Jurisdiction, document_id: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "circular.txt"
        path.write_text(text, encoding="utf-8")
        pages = extract(path)
    return chunk_pages(pages, DocumentMeta(
        document_id=document_id, jurisdiction=jurisdiction, document_version="2.0",
        approval_status="approved", effective_from=date(2025, 1, 1),
        confidentiality_level="internal"))


def context(jurisdiction: Jurisdiction) -> AccessContext:
    return AccessContext(jurisdiction=jurisdiction, role="credit_analyst",
                         as_of_date=date(2026, 1, 15))


@unittest.skipUnless(QDRANT and EMBEDDER,
                     "set CR_TEST_QDRANT_URL and CR_TEST_OMLX_EMBEDDER")
class DeployedStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = settings()
        cls.index = factory.build_index(cls.config)
        for collection in NAMESPACES.values():
            if cls.index.client.collection_exists(collection):
                cls.index.client.delete_collection(collection)
        cls.chunks = deduplicate(
            ingest(SAMA_CIRCULAR, Jurisdiction.SAMA, "SAMA-CIRC-4")
            + ingest(CBUAE_CIRCULAR, Jurisdiction.CBUAE, "CBUAE-CIRC-9"))
        cls.index.upsert(cls.chunks, namespaces=NAMESPACES)

    @classmethod
    def tearDownClass(cls) -> None:
        for collection in NAMESPACES.values():
            if cls.index.client.collection_exists(collection):
                cls.index.client.delete_collection(collection)

    def _retrieve(self, jurisdiction: Jurisdiction, collection: str):
        retriever = factory.build_retriever(self.config)
        predicate = retriever.predicate_for(context(jurisdiction))
        return retriever.index.search_dense(PARAPHRASE, predicate.__class__(
            **{**predicate.__dict__, "collection": collection}), limit=5)

    def test_the_factory_builds_the_real_backends(self) -> None:
        # Not the offline fallback. This is the assertion that would have caught the
        # empty-in-memory-index defect.
        self.assertIsInstance(factory.build_embedder(self.config), OMLXEmbedder)
        self.assertIsInstance(self.index, QdrantPolicyIndex)

    def test_health_reports_the_real_stack(self) -> None:
        described = factory.describe(self.config)
        self.assertEqual(described["index"], "qdrant")
        self.assertEqual(described["semantic"], "true")
        self.assertEqual(described["embedder"], EMBEDDER)

    def test_the_collection_records_the_embedder_that_filled_it(self) -> None:
        stored = self.index._stored_signature(NAMESPACES["SAMA"])
        self.assertTrue(stored.startswith(EMBEDDER), stored)
        self.assertFalse(stored.startswith("?:"), "fell back to the dimension-only check")

    def test_documents_reach_the_index_as_citable_clauses(self) -> None:
        ids = {chunk.evidence_id for chunk in self.chunks}
        self.assertIn("SAMA-CIRC-4#7.2", ids)
        self.assertIn("CBUAE-CIRC-9#4.1", ids)

    def test_the_answering_clause_ranks_first(self) -> None:
        # The query shares no content word with clause 7.2 and does share surface shape
        # with 9.1, so this fails under the hashing placeholder.
        hits = self._retrieve(Jurisdiction.SAMA, NAMESPACES["SAMA"])
        self.assertEqual(hits[0][0].evidence_id, "SAMA-CIRC-4#7.2")

    def test_a_sama_query_never_sees_cbuae_text(self) -> None:
        hits = self._retrieve(Jurisdiction.SAMA, NAMESPACES["SAMA"])
        for chunk, _ in hits:
            self.assertEqual(chunk.jurisdiction, Jurisdiction.SAMA)

    def test_a_cbuae_query_never_sees_sama_text(self) -> None:
        hits = self._retrieve(Jurisdiction.CBUAE, NAMESPACES["CBUAE"])
        self.assertTrue(hits)
        for chunk, _ in hits:
            self.assertEqual(chunk.jurisdiction, Jurisdiction.CBUAE)


if __name__ == "__main__":
    unittest.main()
