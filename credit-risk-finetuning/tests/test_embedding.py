"""The embedder contract, and the two ways a swapped embedder used to corrupt retrieval.

Vectors from different models are not comparable. The failure is silent - the search still
returns a top-k with high cosine scores - so it has to be caught structurally rather than
noticed in an eval.
"""
import os
import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from credit_risk.rag.embedding import (
    DEFAULT_QUERY_INSTRUCTION,
    Embedder,
    HashingEmbedder,
    MLXEmbedder,
    cosine,
)
from credit_risk.rag.filters import RetrievalPolicy
from credit_risk.rag.index import (
    EmbedderMismatch,
    InMemoryPolicyIndex,
    stable_point_id,
)
from credit_risk.rag.schemas import AccessContext, PolicyChunk
from credit_risk.schemas import Jurisdiction

POLICY = Path(__file__).parents[1] / "configs" / "retrieval.yaml"
NAMESPACES = {"SAMA": "policy_sama", "CBUAE": "policy_cbuae"}


def chunk(chunk_id: str, text: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=chunk_id, jurisdiction=Jurisdiction.SAMA, document_id="SAMA-DOC-1",
        document_version="1.0", section_id="7.2", approval_status="approved",
        confidentiality_level="internal", text=text,
    )


def predicate():
    return RetrievalPolicy(POLICY).build_predicate(AccessContext(
        jurisdiction=Jurisdiction.SAMA, role="credit_analyst",
        as_of_date=date(2026, 1, 15)))


class WideHashingEmbedder(HashingEmbedder):
    """Same family, different width - the cheapest way to stage a mismatch offline."""

    model_id = "placeholder-hashing-wide"


class EmbedderContractTests(unittest.TestCase):
    def test_the_placeholder_satisfies_the_protocol(self) -> None:
        self.assertIsInstance(HashingEmbedder(), Embedder)

    def test_the_mlx_embedder_satisfies_the_protocol(self) -> None:
        # Constructing it must not load weights, or configuration code blocks on a
        # multi-hundred-megabyte download.
        self.assertIsInstance(MLXEmbedder(), Embedder)

    def test_the_signature_distinguishes_model_and_width(self) -> None:
        self.assertNotEqual(HashingEmbedder(256).signature, HashingEmbedder(512).signature)
        self.assertNotEqual(
            HashingEmbedder(256).signature, WideHashingEmbedder(256).signature)

    def test_vectors_are_unit_length(self) -> None:
        vector = HashingEmbedder().embed("expected credit loss staging")
        self.assertAlmostEqual(cosine(vector, vector), 1.0, places=6)

    def test_the_placeholder_cannot_rank_a_paraphrase(self) -> None:
        # Pins the limitation as an executable fact rather than a claim in a docstring.
        # The query shares no content word with the clause that answers it, and shares
        # surface shape with one that does not, so a hashed bag of words ranks the wrong
        # clause first. tests/test_embedding_live.py asserts Qwen3-Embedding gets it right.
        sicr = ("A significant increase in credit risk since initial recognition requires "
                "the exposure to be reclassified from stage 1 to stage 2.")
        collateral = ("Eligible financial collateral is revalued at least quarterly and "
                      "the haircut applied follows the supervisory schedule.")
        embedder = HashingEmbedder()
        query = embedder.embed_query(
            "At what point must a loan be downgraded for deteriorating creditworthiness?")
        self.assertLess(cosine(query, embedder.embed(sicr)),
                        cosine(query, embedder.embed(collateral)))

    def test_the_mlx_query_encoding_carries_the_instruction(self) -> None:
        embedder = MLXEmbedder()
        self.assertIn("Instruct:", f"Instruct: {embedder.query_instruction}")
        self.assertEqual(embedder.query_instruction, DEFAULT_QUERY_INSTRUCTION)

    def test_the_mlx_embedder_reports_a_usable_error_without_mlx(self) -> None:
        # On a non-Darwin runner the import fails; the message must name the fix rather
        # than surfacing a bare ImportError from deep in the stack. The import is blocked
        # rather than the real one attempted: a unit test must never pull weights.
        embedder = MLXEmbedder()
        with (
            mock.patch.dict(sys.modules, {"mlx_lm": None}),
            self.assertRaises(RuntimeError) as caught,
        ):
            embedder._load()
        self.assertIn("mlx-lm", str(caught.exception))

    def test_constructing_the_mlx_embedder_loads_nothing(self) -> None:
        # The download this guards against once ran inside the unit suite.
        embedder = MLXEmbedder()
        self.assertIsNone(embedder._model)
        self.assertIsNone(embedder._tokenizer)


class StablePointIdTests(unittest.TestCase):
    def test_the_point_id_is_stable_across_processes(self) -> None:
        # hash() is salted per interpreter, so an id built from it changed on every run
        # and re-ingesting a document inserted duplicates instead of updating.
        script = (
            "from credit_risk.rag.index import stable_point_id;"
            "print(stable_point_id('SAMA-CIRC-4:7.2:0'))"
        )
        env = dict(os.environ, PYTHONHASHSEED="1")
        first = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env,
            check=True).stdout.strip()
        env["PYTHONHASHSEED"] = "2"
        second = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env,
            check=True).stdout.strip()
        self.assertEqual(first, second)
        self.assertEqual(int(first), stable_point_id("SAMA-CIRC-4:7.2:0"))

    def test_different_chunks_get_different_ids(self) -> None:
        self.assertNotEqual(
            stable_point_id("SAMA-CIRC-4:7.2:0"), stable_point_id("SAMA-CIRC-4:7.2:1"))

    def test_the_id_fits_the_qdrant_range(self) -> None:
        self.assertLess(stable_point_id("SAMA-CIRC-4:7.2:0"), 2**63)


class EmbedderMismatchTests(unittest.TestCase):
    def _filled_index(self) -> InMemoryPolicyIndex:
        index = InMemoryPolicyIndex(HashingEmbedder(256))
        index.upsert([chunk("s1", "Significant increase in credit risk means stage 2.")],
                     namespaces=NAMESPACES)
        return index

    def test_searching_with_a_different_embedder_is_refused(self) -> None:
        index = self._filled_index()
        index.embedder = WideHashingEmbedder(256)
        with self.assertRaises(EmbedderMismatch):
            index.search_dense("stage 2", predicate(), limit=3)

    def test_a_different_width_is_refused(self) -> None:
        index = self._filled_index()
        index.embedder = HashingEmbedder(512)
        with self.assertRaises(EmbedderMismatch):
            index.search_dense("stage 2", predicate(), limit=3)

    def test_writing_with_a_different_embedder_is_refused(self) -> None:
        # Mixing vectors inside one collection is worse than querying it wrongly: the
        # collection is then permanently incoherent.
        index = self._filled_index()
        index.embedder = WideHashingEmbedder(256)
        with self.assertRaises(EmbedderMismatch):
            index.upsert([chunk("s2", "Stage 3 provisioning.")], namespaces=NAMESPACES)

    def test_the_matching_embedder_still_searches(self) -> None:
        index = self._filled_index()
        results = index.search_dense("stage 2 credit risk", predicate(), limit=3)
        self.assertTrue(results)

    def test_the_error_names_the_collection_and_both_signatures(self) -> None:
        index = self._filled_index()
        index.embedder = WideHashingEmbedder(256)
        with self.assertRaises(EmbedderMismatch) as caught:
            index.search_dense("stage 2", predicate(), limit=3)
        message = str(caught.exception)
        self.assertIn("policy_sama", message)
        self.assertIn("placeholder-hashing-v1", message)
        self.assertIn("placeholder-hashing-wide", message)


if __name__ == "__main__":
    unittest.main()
