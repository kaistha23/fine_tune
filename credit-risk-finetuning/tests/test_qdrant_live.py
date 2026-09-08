"""Backend parity against a real Qdrant server.

Skipped unless CR_TEST_QDRANT_URL is set, so the offline test container stays green. The
point is that a security control cannot hold in the in-memory reference and quietly fail
in the backend that actually runs in production.

    docker run --rm -d -p 6399:6333 qdrant/qdrant:v1.13.2
    CR_TEST_QDRANT_URL=http://127.0.0.1:6399 uv run python -m unittest tests.test_qdrant_live
"""
import os
import unittest
from datetime import date
from pathlib import Path

from credit_risk.rag.filters import RetrievalPolicy, chunk_is_visible
from credit_risk.rag.index import InMemoryPolicyIndex
from credit_risk.rag.schemas import AccessContext, PolicyChunk
from credit_risk.schemas import Jurisdiction

URL = os.environ.get("CR_TEST_QDRANT_URL")
POLICY = Path(__file__).parents[1] / "configs" / "retrieval.yaml"
NS = {"SAMA": "test_sama", "CBUAE": "test_cbuae"}


def chunk(cid: str, j: Jurisdiction, text: str, **kw) -> PolicyChunk:
    base = dict(chunk_id=cid, jurisdiction=j, document_id=f"{j.value}-DOC-{cid}",
                document_version="1.0", section_id="7.2", approval_status="approved",
                confidentiality_level="internal", text=text)
    base.update(kw)
    return PolicyChunk(**base)


def corpus() -> list[PolicyChunk]:
    return [
        chunk("s1", Jurisdiction.SAMA, "Significant increase in credit risk needs stage 2."),
        chunk("s2", Jurisdiction.SAMA, "Expected credit loss provisioning for stage 3."),
        chunk("s3", Jurisdiction.SAMA, "Draft staging guidance.", approval_status="draft"),
        chunk("s4", Jurisdiction.SAMA, "Restricted annex.",
              confidentiality_level="restricted"),
        chunk("s5", Jurisdiction.SAMA, "Superseded circular.",
              effective_to=date(2024, 12, 31)),
        chunk("s6", Jurisdiction.SAMA, "Future circular.", effective_from=date(2027, 1, 1)),
        chunk("c1", Jurisdiction.CBUAE, "Significant increase in credit risk needs stage 2."),
    ]


@unittest.skipUnless(URL, "set CR_TEST_QDRANT_URL to run against a live Qdrant")
class QdrantParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from credit_risk.rag.index import QdrantPolicyIndex

        cls.index = QdrantPolicyIndex(URL)
        cls.index.upsert(corpus(), namespaces=NS)
        cls.policy = RetrievalPolicy(POLICY)
        base = cls.policy.build_predicate(AccessContext(
            jurisdiction=Jurisdiction.SAMA, role="credit_analyst",
            as_of_date=date(2026, 1, 15)))
        cls.predicate = type(base)(**{**base.__dict__, "collection": NS["SAMA"]})
        cls.expected = {c.chunk_id for c in corpus()
                        if chunk_is_visible(c, cls.predicate)}

    @classmethod
    def tearDownClass(cls) -> None:
        for name in NS.values():
            try:
                cls.index.client.delete_collection(name)
            except Exception:
                pass

    def test_server_side_filter_matches_the_reference_exactly(self) -> None:
        returned = {c.chunk_id for c, _ in
                    self.index.search_dense("credit risk staging", self.predicate, 50)}
        self.assertEqual(returned, self.expected)

    def test_both_backends_return_the_same_set(self) -> None:
        memory = InMemoryPolicyIndex()
        memory.upsert(corpus(), namespaces=NS)
        live = {c.chunk_id for c, _ in
                self.index.search_dense("credit risk staging", self.predicate, 50)}
        ref = {c.chunk_id for c, _ in
               memory.search_dense("credit risk staging", self.predicate, 50)}
        self.assertEqual(live, ref)

    def test_no_cbuae_chunk_can_surface_from_the_sama_collection(self) -> None:
        for arm in (self.index.search_dense, self.index.search_lexical):
            found = arm("credit risk staging", self.predicate, 50)
            for chunk_out, _ in found:
                self.assertEqual(chunk_out.jurisdiction, Jurisdiction.SAMA)

    def test_lexical_arm_respects_the_same_filter(self) -> None:
        returned = {c.chunk_id for c, _ in
                    self.index.search_lexical("credit risk staging", self.predicate, 50)}
        self.assertTrue(returned <= self.expected)


if __name__ == "__main__":
    unittest.main()
