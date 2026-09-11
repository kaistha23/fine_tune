"""Backend parity against a real Qdrant server.

Skipped unless CR_TEST_QDRANT_URL is set, so the offline test container stays green. The
point is that a security control cannot hold in the in-memory reference and quietly fail
in the backend that actually runs in production.

    docker run --rm -d -p 6399:6333 qdrant/qdrant:v1.19.1
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
    base = {
        "chunk_id": cid,
        "jurisdiction": j,
        "document_id": f"{j.value}-DOC-{cid}",
        "document_version": "1.0",
        "section_id": "7.2",
        "approval_status": "approved",
        "confidentiality_level": "internal",
        "text": text,
    }
    base.update(kw)
    return PolicyChunk(**base)


def corpus() -> list[PolicyChunk]:
    return [
        chunk("s1", Jurisdiction.SAMA, "Significant increase in credit risk needs stage 2."),
        chunk("s2", Jurisdiction.SAMA, "Expected credit loss provisioning for stage 3."),
        chunk("s3", Jurisdiction.SAMA, "Draft staging guidance.", approval_status="draft"),
        chunk("s4", Jurisdiction.SAMA, "Restricted annex.", confidentiality_level="restricted"),
        chunk("s5", Jurisdiction.SAMA, "Superseded circular.", effective_to=date(2024, 12, 31)),
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
        base = cls.policy.build_predicate(
            AccessContext(
                jurisdiction=Jurisdiction.SAMA, role="credit_analyst", as_of_date=date(2026, 1, 15)
            )
        )
        cls.predicate = type(base)(**{**base.__dict__, "collection": NS["SAMA"]})
        cls.expected = {c.chunk_id for c in corpus() if chunk_is_visible(c, cls.predicate)}

    @classmethod
    def tearDownClass(cls) -> None:
        for name in NS.values():
            if cls.index.client.collection_exists(name):
                cls.index.client.delete_collection(name)

    def test_server_side_filter_matches_the_reference_exactly(self) -> None:
        returned = {
            c.chunk_id
            for c, _ in self.index.search_dense("credit risk staging", self.predicate, 50)
        }
        self.assertEqual(returned, self.expected)

    def test_both_backends_return_the_same_set(self) -> None:
        memory = InMemoryPolicyIndex()
        memory.upsert(corpus(), namespaces=NS)
        live = {
            c.chunk_id
            for c, _ in self.index.search_dense("credit risk staging", self.predicate, 50)
        }
        ref = {
            c.chunk_id for c, _ in memory.search_dense("credit risk staging", self.predicate, 50)
        }
        self.assertEqual(live, ref)

    def test_no_cbuae_chunk_can_surface_from_the_sama_collection(self) -> None:
        for arm in (self.index.search_dense, self.index.search_lexical):
            found = arm("credit risk staging", self.predicate, 50)
            for chunk_out, _ in found:
                self.assertEqual(chunk_out.jurisdiction, Jurisdiction.SAMA)

    def test_lexical_arm_respects_the_same_filter(self) -> None:
        returned = {
            c.chunk_id
            for c, _ in self.index.search_lexical("credit risk staging", self.predicate, 50)
        }
        self.assertTrue(returned <= self.expected)


def portfolio_corpus() -> list[PolicyChunk]:
    return [
        # Most regulatory guidance names no portfolio: it applies to all of them.
        chunk("p_all", Jurisdiction.SAMA, "Stage 2 on a significant increase in risk."),
        chunk(
            "p_corporate",
            Jurisdiction.SAMA,
            "Corporate obligor staging annex.",
            portfolio=["corporate"],
        ),
        chunk("p_retail", Jurisdiction.SAMA, "Retail staging annex.", portfolio=["retail"]),
    ]


@unittest.skipUnless(URL, "set CR_TEST_QDRANT_URL to run against a live Qdrant")
class PortfolioScopedParityTests(unittest.TestCase):
    """Parity under a portfolio-scoped predicate.

    The other parity class queries with no portfolio, so the portfolio condition was never
    exercised and the two backends diverged unnoticed: chunk_is_visible treats an empty
    portfolio list as "applies to all", while the Qdrant filter's bare MatchAny excluded
    it. Every unscoped circular - which is most of them - was invisible to every
    portfolio-scoped query, and the API returned INSUFFICIENT_EVIDENCE against a full
    index. Found by running the assembled stack, not by any component test.
    """

    COLLECTION = "test_portfolio_sama"

    @classmethod
    def setUpClass(cls) -> None:
        from credit_risk.rag.index import QdrantPolicyIndex

        cls.index = QdrantPolicyIndex(URL)
        if cls.index.client.collection_exists(cls.COLLECTION):
            cls.index.client.delete_collection(cls.COLLECTION)
        cls.index.upsert(portfolio_corpus(), collection=cls.COLLECTION)
        base = RetrievalPolicy(POLICY).build_predicate(
            AccessContext(
                jurisdiction=Jurisdiction.SAMA,
                role="credit_analyst",
                as_of_date=date(2026, 1, 15),
                portfolio="corporate",
            )
        )
        cls.predicate = type(base)(**{**base.__dict__, "collection": cls.COLLECTION})

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.index.client.collection_exists(cls.COLLECTION):
            cls.index.client.delete_collection(cls.COLLECTION)

    def _qdrant_ids(self) -> set[str]:
        found = self.index.search_dense("staging", self.predicate, limit=10)
        return {c.chunk_id for c, _ in found}

    def test_guidance_with_no_portfolio_stays_visible(self) -> None:
        self.assertIn("p_all", self._qdrant_ids())

    def test_guidance_for_the_queried_portfolio_is_visible(self) -> None:
        self.assertIn("p_corporate", self._qdrant_ids())

    def test_guidance_for_another_portfolio_is_hidden(self) -> None:
        self.assertNotIn("p_retail", self._qdrant_ids())

    def test_the_server_agrees_with_the_reference(self) -> None:
        expected = {c.chunk_id for c in portfolio_corpus() if chunk_is_visible(c, self.predicate)}
        self.assertEqual(self._qdrant_ids(), expected)

    def test_both_backends_return_the_same_set(self) -> None:
        memory = InMemoryPolicyIndex()
        memory.upsert(portfolio_corpus(), collection=self.COLLECTION)
        in_memory = {
            c.chunk_id for c, _ in memory.search_dense("staging", self.predicate, limit=10)
        }
        self.assertEqual(self._qdrant_ids(), in_memory)


@unittest.skipUnless(URL, "set CR_TEST_QDRANT_URL to run against a live Qdrant")
class EmbedderSignatureLiveTests(unittest.TestCase):
    """The signature has to survive a round trip through the server.

    Qdrant returns collection metadata under `config`, not at the top level. Reading the
    wrong attribute fell through to a dimension-only fallback that accepts a different
    model of the same width - which is the swap the guard exists to stop, and it looked
    like it was working.
    """

    COLLECTION = "test_signature"

    def setUp(self) -> None:
        from credit_risk.rag.embedding import HashingEmbedder
        from credit_risk.rag.index import QdrantPolicyIndex

        self.index = QdrantPolicyIndex(URL, HashingEmbedder(256))
        if self.index.client.collection_exists(self.COLLECTION):
            self.index.client.delete_collection(self.COLLECTION)
        self.index.upsert(
            [chunk("s1", Jurisdiction.SAMA, "Stage 2 on SICR.")], collection=self.COLLECTION
        )

    def tearDown(self) -> None:
        if self.index.client.collection_exists(self.COLLECTION):
            self.index.client.delete_collection(self.COLLECTION)

    def test_the_signature_round_trips_through_the_server(self) -> None:
        self.assertEqual(
            self.index._stored_signature(self.COLLECTION), "placeholder-hashing-v1:256"
        )

    def test_a_different_model_at_the_same_width_is_refused(self) -> None:
        from credit_risk.rag.embedding import HashingEmbedder
        from credit_risk.rag.index import EmbedderMismatch, QdrantPolicyIndex

        class Wide(HashingEmbedder):
            model_id = "placeholder-hashing-wide"

        other = QdrantPolicyIndex(URL, Wide(256))
        with self.assertRaises(EmbedderMismatch):
            other.ensure_collection(self.COLLECTION)

    def test_a_different_width_is_refused(self) -> None:
        from credit_risk.rag.embedding import HashingEmbedder
        from credit_risk.rag.index import EmbedderMismatch, QdrantPolicyIndex

        other = QdrantPolicyIndex(URL, HashingEmbedder(512))
        with self.assertRaises(EmbedderMismatch):
            other.ensure_collection(self.COLLECTION)

    def test_the_matching_embedder_is_accepted(self) -> None:
        from credit_risk.rag.embedding import HashingEmbedder
        from credit_risk.rag.index import QdrantPolicyIndex

        QdrantPolicyIndex(URL, HashingEmbedder(256)).ensure_collection(self.COLLECTION)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(URL, "set CR_TEST_QDRANT_URL to run against a live Qdrant")
class PaginationAndACLTests(unittest.TestCase):
    def test_complete_lexical_corpus_and_document_roles(self):
        from uuid import uuid4

        from credit_risk.rag.index import QdrantPolicyIndex

        index = QdrantPolicyIndex(URL)
        collection = "test_pagination_" + uuid4().hex
        chunks = [
            chunk(f"page-{i}", Jurisdiction.SAMA, f"Credit staging uniqueitem{i}.")
            for i in range(600)
        ]
        chunks.append(
            chunk(
                "restricted-role",
                Jurisdiction.SAMA,
                "Credit staging secretannex.",
                allowed_roles=["regulator_liaison"],
            )
        )
        try:
            index.upsert(chunks, collection=collection)
            base = RetrievalPolicy(POLICY).build_predicate(
                AccessContext(
                    jurisdiction=Jurisdiction.SAMA,
                    role="credit_analyst",
                    as_of_date=date(2026, 1, 15),
                )
            )
            predicate = type(base)(**{**base.__dict__, "collection": collection})
            returned = {
                c.chunk_id for c, _ in index.search_lexical("credit staging", predicate, 700)
            }
            self.assertEqual(returned, {c.chunk_id for c in chunks[:-1]})
            dense = {c.chunk_id for c, _ in index.search_dense("secretannex", predicate, 700)}
            self.assertNotIn("restricted-role", dense)
        finally:
            if index.client.collection_exists(collection):
                index.client.delete_collection(collection)
