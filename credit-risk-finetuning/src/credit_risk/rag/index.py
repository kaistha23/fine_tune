"""Index backends. Both apply the same pre-search predicate.

InMemoryPolicyIndex is the reference: it filters with chunk_is_visible, the same function
the tests assert against. QdrantPolicyIndex translates the predicate into a Qdrant filter
and passes it to the query, so the server never scores an excluded point.

Qdrant is imported lazily. The test container runs on an internal network with no Qdrant
and no qdrant-client installed, and the security properties must still be testable there.
"""
from __future__ import annotations

from typing import Any, Protocol

from credit_risk.rag.embedding import Embedder, HashingEmbedder, cosine
from credit_risk.rag.filters import AccessPredicate, chunk_is_visible
from credit_risk.rag.lexical import BM25
from credit_risk.rag.schemas import PolicyChunk


class PolicyIndex(Protocol):
    def upsert(self, chunks: list[PolicyChunk]) -> None:
        ...

    def search_dense(self, query: str, predicate: AccessPredicate,
                     limit: int) -> list[tuple[PolicyChunk, float]]:
        ...

    def search_lexical(self, query: str, predicate: AccessPredicate,
                       limit: int) -> list[tuple[PolicyChunk, float]]:
        ...


class InMemoryPolicyIndex:
    """Reference backend, partitioned by collection exactly as Qdrant is."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or HashingEmbedder()
        self._collections: dict[str, list[PolicyChunk]] = {}
        self._vectors: dict[str, list[list[float]]] = {}

    def _collection_name(self, chunk: PolicyChunk, namespaces: dict[str, str]) -> str:
        return namespaces[chunk.jurisdiction.value]

    def upsert(self, chunks: list[PolicyChunk], collection: str | None = None,
               namespaces: dict[str, str] | None = None) -> None:
        for chunk in chunks:
            name = collection or (namespaces or {}).get(chunk.jurisdiction.value)
            if name is None:
                raise ValueError(
                    f"No collection for jurisdiction {chunk.jurisdiction.value}; "
                    "a chunk must never land in a shared namespace"
                )
            self._collections.setdefault(name, []).append(chunk)
            self._vectors.setdefault(name, []).append(self.embedder.embed(chunk.text))

    def _visible(self, predicate: AccessPredicate) -> list[tuple[int, PolicyChunk]]:
        chunks = self._collections.get(predicate.collection, [])
        return [(i, c) for i, c in enumerate(chunks) if chunk_is_visible(c, predicate)]

    def search_dense(self, query: str, predicate: AccessPredicate,
                     limit: int) -> list[tuple[PolicyChunk, float]]:
        visible = self._visible(predicate)
        if not visible:
            return []
        vectors = self._vectors[predicate.collection]
        query_vector = self.embedder.embed(query)
        scored = [(chunk, cosine(query_vector, vectors[i])) for i, chunk in visible]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def search_lexical(self, query: str, predicate: AccessPredicate,
                       limit: int) -> list[tuple[PolicyChunk, float]]:
        visible = self._visible(predicate)
        if not visible:
            return []
        from credit_risk.rag.lexical import tokenize
        bm25 = BM25([tokenize(chunk.text) for _, chunk in visible])
        ranked = bm25.rank(query)
        return [(visible[i][1], score) for i, score in ranked[:limit]]


def build_qdrant_filter(predicate: AccessPredicate) -> Any:
    """Translate the predicate into a Qdrant filter applied server-side, pre-search.

    Effective dates are half-open ranges rather than equality, and a chunk with no
    effective_from or effective_to must still match - hence the is-null alternatives.
    """
    from qdrant_client import models

    stamp = predicate.effective_on.isoformat()
    conditions: list[Any] = [
        models.FieldCondition(key="jurisdiction",
                              match=models.MatchValue(value=predicate.jurisdiction)),
        models.FieldCondition(key="approval_status",
                              match=models.MatchAny(any=predicate.approval_status_in)),
        models.FieldCondition(key="confidentiality_level",
                              match=models.MatchAny(any=predicate.confidentiality_in)),
    ]
    if predicate.portfolio:
        conditions.append(
            models.FieldCondition(key="portfolio",
                                  match=models.MatchAny(any=[predicate.portfolio])))
    return models.Filter(
        must=conditions,
        must_not=[
            models.FieldCondition(key="effective_from",
                                  range=models.DatetimeRange(gt=stamp)),
            models.FieldCondition(key="effective_to",
                                  range=models.DatetimeRange(lt=stamp)),
        ],
    )
