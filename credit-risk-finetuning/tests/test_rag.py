"""Jurisdiction isolation and ACL-aware retrieval (finding C2).

The handover repository ran Qdrant as a container with no retrieval code, so SAMA/CBUAE
separation was a post-hoc equality check on evidence nothing retrieved. These tests assert
the separation happens *before* the search.
"""
import unittest
from datetime import date
from pathlib import Path

from credit_risk.rag.filters import AccessPolicyError, RetrievalPolicy, chunk_is_visible
from credit_risk.rag.index import InMemoryPolicyIndex
from credit_risk.rag.retriever import PolicyRetriever
from credit_risk.rag.schemas import AccessContext, PolicyChunk
from credit_risk.schemas import Jurisdiction

POLICY = Path(__file__).parents[1] / "configs" / "retrieval.yaml"
NAMESPACES = {"SAMA": "policy_sama", "CBUAE": "policy_cbuae"}


def chunk(chunk_id: str, jurisdiction: Jurisdiction, text: str, **overrides) -> PolicyChunk:
    base = dict(
        chunk_id=chunk_id, section_key=chunk_id, jurisdiction=jurisdiction,
        document_id=f"{jurisdiction.value}-DOC-1", document_version="1.0",
        section_id="7.2", approval_status="approved",
        confidentiality_level="internal", text=text,
    )
    base.update(overrides)
    return PolicyChunk(**base)


def corpus() -> list[PolicyChunk]:
    return [
        chunk("s1", Jurisdiction.SAMA,
              "Significant increase in credit risk requires stage 2 classification."),
        chunk("s2", Jurisdiction.SAMA,
              "Provisions for stage 3 exposures follow the expected credit loss model."),
        chunk("s3", Jurisdiction.SAMA, "Draft guidance on staging.", approval_status="draft"),
        chunk("s4", Jurisdiction.SAMA, "Restricted supervisory annex on staging.",
              confidentiality_level="restricted"),
        chunk("s5", Jurisdiction.SAMA, "Superseded staging circular.",
              effective_to=date(2024, 12, 31)),
        chunk("s6", Jurisdiction.SAMA, "Future staging circular, not yet effective.",
              effective_from=date(2027, 1, 1)),
        chunk("c1", Jurisdiction.CBUAE,
              "Significant increase in credit risk requires stage 2 classification."),
        chunk("c2", Jurisdiction.CBUAE, "Expected credit loss provisioning rules."),
    ]


def retriever() -> PolicyRetriever:
    index = InMemoryPolicyIndex()
    index.upsert(corpus(), namespaces=NAMESPACES)
    return PolicyRetriever(RetrievalPolicy(POLICY), index)


def context(**overrides) -> AccessContext:
    base = dict(jurisdiction=Jurisdiction.SAMA, role="credit_analyst",
                as_of_date=date(2026, 1, 15))
    base.update(overrides)
    return AccessContext(**base)


class NamespaceIsolationTests(unittest.TestCase):
    def test_sama_and_cbuae_resolve_to_different_collections(self) -> None:
        policy = RetrievalPolicy(POLICY)
        self.assertNotEqual(policy.collection_for("SAMA"), policy.collection_for("CBUAE"))

    def test_a_sama_query_never_returns_a_cbuae_chunk(self) -> None:
        # The adversarial case both plans call out. The two jurisdictions hold nearly
        # identical text, so a shared index would surface the wrong one on similarity.
        result = retriever().retrieve(
            "significant increase in credit risk staging", context())
        self.assertGreater(result["retrieved"], 0)
        for evidence in result["evidence"]:
            self.assertEqual(evidence.jurisdiction, Jurisdiction.SAMA)
            self.assertNotIn("CBUAE", evidence.document_id)

    def test_isolation_is_structural_not_a_post_filter(self) -> None:
        # CBUAE chunks are not filtered out of the results; they sit in a different
        # collection, so the SAMA search never scores them at all.
        r = retriever()
        predicate = r.predicate_for(context())
        self.assertEqual(predicate.collection, "policy_sama")
        searched = r.index.search_dense("credit loss", predicate, 50)
        self.assertTrue(all(c.jurisdiction == Jurisdiction.SAMA for c, _ in searched))

    def test_unknown_jurisdiction_has_no_collection(self) -> None:
        with self.assertRaises(AccessPolicyError):
            RetrievalPolicy(POLICY).collection_for("ECB")


class AccessControlTests(unittest.TestCase):
    def test_unknown_role_sees_nothing(self) -> None:
        with self.assertRaises(AccessPolicyError):
            retriever().retrieve("staging", context(role="intern"))

    def test_analyst_cannot_reach_restricted_material(self) -> None:
        result = retriever().retrieve("restricted supervisory annex", context())
        for evidence in result["evidence"]:
            self.assertNotIn("Restricted supervisory", evidence.text)

    def test_senior_officer_reaches_confidential_material(self) -> None:
        policy = RetrievalPolicy(POLICY)
        self.assertNotIn("confidential", policy.levels_for_role("credit_analyst"))
        self.assertIn("confidential", policy.levels_for_role("senior_credit_officer"))

    def test_draft_documents_are_invisible(self) -> None:
        result = retriever().retrieve("draft guidance on staging", context())
        for evidence in result["evidence"]:
            self.assertNotIn("Draft guidance", evidence.text)


class EffectiveDateTests(unittest.TestCase):
    def test_superseded_and_not_yet_effective_are_excluded(self) -> None:
        result = retriever().retrieve("staging circular", context())
        texts = " ".join(e.text for e in result["evidence"])
        self.assertNotIn("Superseded", texts)
        self.assertNotIn("not yet effective", texts)

    def test_asking_as_at_an_earlier_date_changes_what_is_visible(self) -> None:
        superseded = chunk("s5", Jurisdiction.SAMA, "old", effective_to=date(2024, 12, 31))
        now = RetrievalPolicy(POLICY).build_predicate(context())
        self.assertFalse(chunk_is_visible(superseded, now))
        then = RetrievalPolicy(POLICY).build_predicate(context(as_of_date=date(2024, 6, 30)))
        self.assertTrue(chunk_is_visible(superseded, then))


class HybridRetrievalTests(unittest.TestCase):
    def test_both_arms_return_candidates(self) -> None:
        r = retriever()
        predicate = r.predicate_for(context())
        self.assertTrue(r.index.search_dense("expected credit loss", predicate, 10))
        self.assertTrue(r.index.search_lexical("expected credit loss", predicate, 10))

    def test_empty_retrieval_reports_insufficient_evidence(self) -> None:
        empty = PolicyRetriever(RetrievalPolicy(POLICY), InMemoryPolicyIndex())
        result = empty.retrieve("anything at all", context())
        self.assertEqual(result["answer_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["retrieved"], 0)

    def test_access_filter_is_reported_for_audit(self) -> None:
        joined = " ".join(retriever().retrieve("staging", context())["access_filter"])
        self.assertIn("collection=policy_sama", joined)
        self.assertIn("jurisdiction=SAMA", joined)
        self.assertIn("approved", joined)


class BackendParityTests(unittest.TestCase):
    def test_a_chunk_cannot_be_indexed_without_a_namespace(self) -> None:
        with self.assertRaises(ValueError):
            InMemoryPolicyIndex().upsert([chunk("x", Jurisdiction.SAMA, "text")])

    def test_jurisdiction_check_still_holds_if_a_chunk_is_misfiled(self) -> None:
        # Defence in depth: even with a CBUAE chunk wrongly placed in the SAMA
        # collection, the predicate rejects it on jurisdiction.
        index = InMemoryPolicyIndex()
        index.upsert(corpus(), namespaces=NAMESPACES)
        index.upsert([chunk("leak", Jurisdiction.CBUAE, "leaked staging text")],
                     collection="policy_sama")
        result = PolicyRetriever(RetrievalPolicy(POLICY), index).retrieve(
            "leaked staging text", context())
        for evidence in result["evidence"]:
            self.assertEqual(evidence.jurisdiction, Jurisdiction.SAMA)


if __name__ == "__main__":
    unittest.main()
