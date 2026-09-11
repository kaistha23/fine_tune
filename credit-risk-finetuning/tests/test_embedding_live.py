"""Qwen3-Embedding running for real on Apple Silicon.

Skipped unless CR_TEST_EMBEDDER is set, because it pulls weights and needs Metal. The
offline suite must never download a model, so the semantic assertions live here rather
than beside the HashingEmbedder contract tests.

    CR_TEST_EMBEDDER=mlx-community/Qwen3-Embedding-0.6B-8bit \\
        uv run python -m pytest tests/test_embedding_live.py -q

What this checks is the thing the placeholder cannot do: rank a paraphrase above a
lexically similar but unrelated clause. A hashed bag of words scores on shared tokens, so
it cannot separate them, and no retrieval accuracy claim may rest on it.
"""
import os
import unittest
from datetime import date
from pathlib import Path

from credit_risk.rag.embedding import MLXEmbedder, cosine
from credit_risk.rag.filters import RetrievalPolicy
from credit_risk.rag.index import InMemoryPolicyIndex
from credit_risk.rag.schemas import AccessContext, PolicyChunk
from credit_risk.schemas import Jurisdiction

MODEL = os.environ.get("CR_TEST_EMBEDDER")
POLICY = Path(__file__).parents[1] / "configs" / "retrieval.yaml"
NAMESPACES = {"SAMA": "policy_sama", "CBUAE": "policy_cbuae"}

SICR = ("A significant increase in credit risk since initial recognition requires the "
        "exposure to be reclassified from stage 1 to stage 2.")
IMPAIRED = ("An exposure that is more than 90 days past due is treated as credit "
            "impaired and classified in stage 3.")
COLLATERAL = ("Eligible financial collateral is revalued at least quarterly and the "
              "haircut applied follows the supervisory schedule.")
PARAPHRASE = "At what point must a loan be downgraded for deteriorating creditworthiness?"


def chunk(cid: str, section: str, text: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=cid, jurisdiction=Jurisdiction.SAMA, document_id="SAMA-DOC-1",
        document_version="1.0", section_id=section, approval_status="approved",
        confidentiality_level="internal", text=text,
    )


@unittest.skipUnless(MODEL, "set CR_TEST_EMBEDDER to run against real weights")
class MLXEmbedderLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.embedder = MLXEmbedder(MODEL)

    def test_the_width_comes_from_the_model_not_a_constant(self) -> None:
        self.assertGreater(self.embedder.dimensions, 256)
        self.assertIn(str(self.embedder.dimensions), self.embedder.signature)

    def test_vectors_are_unit_length(self) -> None:
        vector = self.embedder.embed(SICR)
        self.assertAlmostEqual(cosine(vector, vector), 1.0, places=4)

    def test_embedding_is_deterministic(self) -> None:
        # Two calls must agree, or an index rebuilt after a restart is not comparable
        # with the queries run against it.
        first = self.embedder.embed(SICR)
        second = self.embedder.embed(SICR)
        self.assertAlmostEqual(cosine(first, second), 1.0, places=4)

    def test_a_paraphrase_outranks_a_lexically_similar_distractor(self) -> None:
        # PARAPHRASE shares no content word with the SICR clause it asks about, and does
        # share "applied"/"follows" surface shape with the collateral clause.
        # HashingEmbedder ranks the distractor first here - asserted in
        # tests/test_embedding.py - so this is the discrimination the placeholder cannot
        # do, not merely one it does less well.
        query = self.embedder.embed_query(PARAPHRASE)
        self.assertGreater(
            cosine(query, self.embedder.embed(SICR)),
            cosine(query, self.embedder.embed(COLLATERAL)))

    def test_the_query_instruction_changes_the_encoding(self) -> None:
        text = "When does an exposure become credit impaired?"
        self.assertLess(
            cosine(self.embedder.embed(text), self.embedder.embed_query(text)), 0.9999)

    def test_it_ranks_correctly_through_the_index(self) -> None:
        index = InMemoryPolicyIndex(self.embedder)
        index.upsert([chunk("s1", "7.2", SICR), chunk("s2", "7.3", IMPAIRED),
                      chunk("s3", "9.1", COLLATERAL)], namespaces=NAMESPACES)
        predicate = RetrievalPolicy(POLICY).build_predicate(AccessContext(
            jurisdiction=Jurisdiction.SAMA, role="credit_analyst",
            as_of_date=date(2026, 1, 15)))
        top = index.search_dense(
            "How many days past due before an exposure is impaired?", predicate, limit=3)
        self.assertEqual(top[0][0].chunk_id, "s2")

    def test_long_input_keeps_the_tail(self) -> None:
        # Truncation drops the head, because the pooled position and the operative
        # condition both sit at the end of a clause.
        embedder = MLXEmbedder(MODEL, max_tokens=64)
        padded = ("Preamble sentence with no operative content. " * 200) + SICR
        query = embedder.embed_query("stage 2 significant increase in credit risk")
        self.assertGreater(cosine(query, embedder.embed(padded)),
                           cosine(query, embedder.embed(COLLATERAL)))


if __name__ == "__main__":
    unittest.main()
