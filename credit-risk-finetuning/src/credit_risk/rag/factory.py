"""Build the retrieval stack from settings.

compose.yaml has always set CR_QDRANT_URL and run a Qdrant service, but Settings had no
field for it and `extra="ignore"` dropped it. `api.py` constructed PolicyRetriever with no
index, which defaults to an empty InMemoryPolicyIndex, so the deployed API retrieved
nothing from a database it was wired to and never opened. This is where that gets decided
once, from configuration, instead of by a default argument.

The choice is deliberately explicit rather than "use Qdrant if it is reachable": a
retrieval stack that silently degrades to an empty in-memory index answers questions with
no evidence at all, and the guardrails then report low coverage rather than a broken
backend.
"""
from __future__ import annotations

from credit_risk.rag.embedding import Embedder, HashingEmbedder, OMLXEmbedder
from credit_risk.rag.filters import RetrievalPolicy
from credit_risk.rag.index import InMemoryPolicyIndex, PolicyIndex
from credit_risk.rag.retriever import PolicyRetriever
from credit_risk.settings import Settings


def build_embedder(config: Settings) -> Embedder:
    """The oMLX embedder when a model is configured, otherwise the placeholder.

    MLXEmbedder is deliberately not reachable from here. It needs Metal, and the API runs
    in a Linux container; offering it as a setting would produce a stack that works on the
    developer's Mac and fails on deployment.
    """
    if config.embedding_model:
        return OMLXEmbedder(
            base_url=config.omlx_base_url,
            model=config.embedding_model,
            api_key=config.omlx_api_key,
        )
    return HashingEmbedder()


def build_index(config: Settings, embedder: Embedder | None = None) -> PolicyIndex:
    embedder = embedder or build_embedder(config)
    if config.qdrant_url:
        from credit_risk.rag.index import QdrantPolicyIndex
        return QdrantPolicyIndex(config.qdrant_url, embedder)
    return InMemoryPolicyIndex(embedder)


def build_retriever(config: Settings, policy: RetrievalPolicy | None = None
                    ) -> PolicyRetriever:
    policy = policy or RetrievalPolicy(config.retrieval_policy)
    return PolicyRetriever(policy, build_index(config))


def describe(config: Settings) -> dict[str, str]:
    """What the retrieval stack actually is, for /health.

    An operator has to be able to tell a real backend from the offline fallback without
    reading the container's environment, because the fallback answers every question with
    no evidence and looks like a quiet corpus.
    """
    return {
        "index": "qdrant" if config.qdrant_url else "in-memory",
        "embedder": config.embedding_model or "placeholder-hashing-v1",
        "semantic": "true" if config.embedding_model else "false",
    }
