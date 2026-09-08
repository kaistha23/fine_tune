"""Document chunking and lineage (the missing producer for the RAG index).

The retrieval layer consumed PolicyChunk records but nothing produced them.
"""
import unittest
from datetime import date
from pathlib import Path

from credit_risk.rag.filters import RetrievalPolicy, chunk_is_visible
from credit_risk.rag.ingest import (
    MAX_CHARS,
    DocumentMeta,
    chunk_document,
    deduplicate,
    split_sections,
)
from credit_risk.rag.schemas import AccessContext
from credit_risk.schemas import Jurisdiction

POLICY = Path(__file__).parents[1] / "configs" / "retrieval.yaml"

CIRCULAR = """1. Scope

This circular applies to all licensed banks operating in the Kingdom.

7. Staging

7.1 Initial recognition

Exposures are classified in stage 1 on initial recognition.

7.2 Significant increase in credit risk

A significant increase in credit risk requires reclassification to stage 2.
The assessment considers days past due, watchlist status and rating migration.

7.3 Credit impaired

An exposure more than 90 days past due is classified in stage 3.
"""


def meta(**overrides) -> DocumentMeta:
    base = dict(
        document_id="SAMA-CIRC-4", jurisdiction=Jurisdiction.SAMA,
        document_version="2.0", authority="SAMA", document_type="circular",
        approval_status="approved", effective_from=date(2025, 1, 1),
        confidentiality_level="internal",
    )
    base.update(overrides)
    return DocumentMeta(**base)


class SectionSplittingTests(unittest.TestCase):
    def test_numbered_clauses_become_their_own_sections(self) -> None:
        ids = [s.section_id for s in split_sections(CIRCULAR)]
        for expected in ("1", "7.1", "7.2", "7.3"):
            self.assertIn(expected, ids)

    def test_heading_path_records_where_a_clause_sits(self) -> None:
        sections = {s.section_id: s for s in split_sections(CIRCULAR)}
        path = sections["7.2"].heading_path
        self.assertTrue(any("Staging" in part for part in path))
        self.assertTrue(any("Significant increase" in part for part in path))

    def test_a_clause_keeps_its_qualifying_sentence(self) -> None:
        # Splitting mid-clause strips the condition off the rule, which is how a model
        # ends up asserting an obligation without its exception.
        sections = {s.section_id: s for s in split_sections(CIRCULAR)}
        body = sections["7.2"].text
        self.assertIn("reclassification to stage 2", body)
        self.assertIn("days past due, watchlist status", body)


class ChunkLineageTests(unittest.TestCase):
    def test_every_chunk_carries_the_document_lineage(self) -> None:
        for chunk in chunk_document(CIRCULAR, meta()):
            self.assertEqual(chunk.document_id, "SAMA-CIRC-4")
            self.assertEqual(chunk.document_version, "2.0")
            self.assertEqual(chunk.jurisdiction, Jurisdiction.SAMA)
            self.assertEqual(chunk.approval_status, "approved")
            self.assertTrue(chunk.content_hash)

    def test_evidence_id_is_a_citable_clause_reference(self) -> None:
        chunks = {c.section_id: c for c in chunk_document(CIRCULAR, meta())}
        self.assertEqual(chunks["7.2"].evidence_id, "SAMA-CIRC-4#7.2")

    def test_chunks_respect_the_size_ceiling(self) -> None:
        long_text = "5. Long clause\n\n" + ("A sentence about provisioning. " * 400)
        for chunk in chunk_document(long_text, meta()):
            self.assertLessEqual(len(chunk.text), MAX_CHARS)

    def test_ingested_chunks_are_immediately_filterable(self) -> None:
        # An ingested chunk must satisfy the same predicate the retriever applies, or the
        # producer and the consumer disagree.
        predicate = RetrievalPolicy(POLICY).build_predicate(AccessContext(
            jurisdiction=Jurisdiction.SAMA, role="credit_analyst",
            as_of_date=date(2026, 1, 15)))
        chunks = chunk_document(CIRCULAR, meta())
        self.assertTrue(any(chunk_is_visible(c, predicate) for c in chunks))

    def test_a_draft_document_produces_invisible_chunks(self) -> None:
        predicate = RetrievalPolicy(POLICY).build_predicate(AccessContext(
            jurisdiction=Jurisdiction.SAMA, role="credit_analyst",
            as_of_date=date(2026, 1, 15)))
        chunks = chunk_document(CIRCULAR, meta(approval_status="draft"))
        self.assertFalse(any(chunk_is_visible(c, predicate) for c in chunks))

    def test_a_superseded_document_produces_invisible_chunks(self) -> None:
        predicate = RetrievalPolicy(POLICY).build_predicate(AccessContext(
            jurisdiction=Jurisdiction.SAMA, role="credit_analyst",
            as_of_date=date(2026, 1, 15)))
        chunks = chunk_document(CIRCULAR, meta(effective_to=date(2025, 6, 30)))
        self.assertFalse(any(chunk_is_visible(c, predicate) for c in chunks))


class DeduplicationTests(unittest.TestCase):
    def test_identical_boilerplate_is_dropped_within_a_jurisdiction(self) -> None:
        chunks = chunk_document(CIRCULAR, meta())
        doubled = chunks + chunk_document(CIRCULAR, meta())
        self.assertEqual(len(deduplicate(doubled)), len(chunks))

    def test_identical_text_in_two_jurisdictions_is_kept(self) -> None:
        # SAMA and CBUAE often say the same thing. Collapsing them would silently make
        # one jurisdiction cite the other's document.
        both = (chunk_document(CIRCULAR, meta())
                + chunk_document(CIRCULAR, meta(document_id="CBUAE-CIRC-9",
                                                jurisdiction=Jurisdiction.CBUAE)))
        self.assertEqual(len(deduplicate(both)), len(both))


if __name__ == "__main__":
    unittest.main()
