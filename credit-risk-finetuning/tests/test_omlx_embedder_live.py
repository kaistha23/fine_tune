"""OMLXEmbedder against a running oMLX server.

This is the embedder the deployed API actually uses - the container has no Metal, so it
asks the native server over HTTP - and it is the one least covered by the offline suite.

Skipped unless CR_TEST_OMLX_EMBEDDER names an embedding model the server has loaded:

    CR_TEST_OMLX_EMBEDDER=qwen3-embedding-0.6b \\
    CR_OMLX_API_KEY=... \\
    CR_TEST_OMLX_URL=http://127.0.0.1:9905/v1 \\
        uv run python -m pytest tests/test_omlx_embedder_live.py -q

The assertions are the same properties the in-process embedder must satisfy, because the
index cannot tell which produced a vector and scores both with a plain dot product.
"""
import os
import unittest

from credit_risk.rag.embedding import OMLXEmbedder, cosine

MODEL = os.environ.get("CR_TEST_OMLX_EMBEDDER")
URL = os.environ.get("CR_TEST_OMLX_URL", "http://127.0.0.1:9905/v1")
API_KEY = os.environ.get("CR_OMLX_API_KEY", "")

SICR = ("A significant increase in credit risk since initial recognition requires the "
        "exposure to be reclassified from stage 1 to stage 2.")
COLLATERAL = ("Eligible financial collateral is revalued at least quarterly and the "
              "haircut applied follows the supervisory schedule.")
PARAPHRASE = "At what point must a loan be downgraded for deteriorating creditworthiness?"


@unittest.skipUnless(MODEL, "set CR_TEST_OMLX_EMBEDDER to run against a live oMLX server")
class OMLXEmbedderLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.embedder = OMLXEmbedder(URL, MODEL, api_key=API_KEY)

    def test_dimensionality_is_discovered_from_the_server(self) -> None:
        # Hard-coding it would let a model change pass unnoticed until the vectors were
        # already in the index.
        self.assertGreater(self.embedder.dimensions, 0)
        self.assertIn(str(self.embedder.dimensions), self.embedder.signature)
        self.assertTrue(self.embedder.signature.startswith(MODEL))

    def test_vectors_are_unit_length(self) -> None:
        # Servers differ on whether they normalise, and InMemoryPolicyIndex scores with a
        # plain dot product, so the client normalises rather than trusting the server.
        vector = self.embedder.embed(SICR)
        self.assertAlmostEqual(cosine(vector, vector), 1.0, places=4)

    def test_the_width_is_consistent_across_calls(self) -> None:
        self.assertEqual(len(self.embedder.embed(SICR)),
                         len(self.embedder.embed(COLLATERAL)))

    def test_embedding_is_deterministic(self) -> None:
        self.assertAlmostEqual(
            cosine(self.embedder.embed(SICR), self.embedder.embed(SICR)), 1.0, places=4)

    def test_the_query_instruction_changes_the_encoding(self) -> None:
        self.assertLess(
            cosine(self.embedder.embed(PARAPHRASE),
                   self.embedder.embed_query(PARAPHRASE)), 0.9999)

    def test_a_paraphrase_outranks_a_lexically_similar_distractor(self) -> None:
        query = self.embedder.embed_query(PARAPHRASE)
        self.assertGreater(cosine(query, self.embedder.embed(SICR)),
                           cosine(query, self.embedder.embed(COLLATERAL)))


if __name__ == "__main__":
    unittest.main()
