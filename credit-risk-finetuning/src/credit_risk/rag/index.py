"""Index backends. Both apply the same pre-search predicate.

InMemoryPolicyIndex is the reference: it filters with chunk_is_visible, the same function
the tests assert against. QdrantPolicyIndex translates the same predicate into a Qdrant
filter passed as query_filter, so the server never scores an excluded point.

Both are checked against the same visibility cases, so a security control cannot hold in
one backend and quietly fail in the other.

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


def to_payload(chunk: PolicyChunk) -> dict[str, Any]:
    """Chunk fields as a Qdrant payload.

    Dates become ISO strings so they can be filtered as datetime ranges server-side. A
    chunk with no effective_from or effective_to simply omits the key, which is what makes
    the must_not range conditions leave open-ended documents visible.
    """
    payload = chunk.model_dump(mode="json")
    for key in ("effective_from", "effective_to"):
        if payload.get(key) is None:
            payload.pop(key, None)
        else:
            payload[key] = f"{payload[key]}T00:00:00Z"
    return payload


def from_payload(payload: dict[str, Any]) -> PolicyChunk:
    data = dict(payload)
    for key in ("effective_from", "effective_to"):
        if data.get(key):
            data[key] = str(data[key])[:10]
    return PolicyChunk(**data)


class QdrantPolicyIndex:
    """Qdrant-backed index, one collection per jurisdiction.

    qdrant_client is imported lazily: the offline test container has neither the package
    nor a server, and the access controls must stay testable there.
    """

    def __init__(self, url: str, embedder: Embedder | None = None,
                 timeout: float = 10.0):
        from qdrant_client import QdrantClient

        self.embedder = embedder or HashingEmbedder()
        self.client = QdrantClient(url=url, timeout=timeout)

    def ensure_collection(self, collection: str) -> None:
        from qdrant_client import models

        if self.client.collection_exists(collection):
            return
        self.client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dimensions, distance=models.Distance.COSINE),
        )
        # Indexing the filtered fields is what keeps the pre-search filter cheap rather
        # than a full scan on every query.
        for field, schema in (
            ("jurisdiction", "keyword"), ("approval_status", "keyword"),
            ("confidentiality_level", "keyword"), ("portfolio", "keyword"),
            ("effective_from", "datetime"), ("effective_to", "datetime"),
        ):
            self.client.create_payload_index(
                collection_name=collection, field_name=field, field_schema=schema)

    def upsert(self, chunks: list[PolicyChunk], collection: str | None = None,
               namespaces: dict[str, str] | None = None) -> None:
        from qdrant_client import models

        grouped: dict[str, list[PolicyChunk]] = {}
        for chunk in chunks:
            name = collection or (namespaces or {}).get(chunk.jurisdiction.value)
            if name is None:
                raise ValueError(
                    f"No collection for jurisdiction {chunk.jurisdiction.value}; "
                    "a chunk must never land in a shared namespace"
                )
            grouped.setdefault(name, []).append(chunk)

        for name, group in grouped.items():
            self.ensure_collection(name)
            self.client.upsert(
                collection_name=name,
                points=[
                    models.PointStruct(
                        id=abs(hash(chunk.chunk_id)) % (2**63),
                        vector=self.embedder.embed(chunk.text),
                        payload=to_payload(chunk),
                    )
                    for chunk in group
                ],
            )

    def search_dense(self, query: str, predicate: AccessPredicate,
                     limit: int) -> list[tuple[PolicyChunk, float]]:
        if not self.client.collection_exists(predicate.collection):
            return []
        found = self.client.query_points(
            collection_name=predicate.collection,
            query=self.embedder.embed(query),
            query_filter=build_qdrant_filter(predicate),
            limit=limit,
            with_payload=True,
        ).points
        return [(from_payload(point.payload), float(point.score)) for point in found]

    def search_lexical(self, query: str, predicate: AccessPredicate,
                       limit: int) -> list[tuple[PolicyChunk, float]]:
        """Candidates come back under the same server-side filter, then BM25 ranks them.

        Qdrant does the access filtering; the lexical scoring is ours, so both backends
        rank identically for a given candidate set.
        """
        from credit_risk.rag.lexical import tokenize

        if not self.client.collection_exists(predicate.collection):
            return []
        points, _ = self.client.scroll(
            collection_name=predicate.collection,
            scroll_filter=build_qdrant_filter(predicate),
            limit=max(limit * 8, 128),
            with_payload=True,
        )
        candidates = [from_payload(point.payload) for point in points]
        if not candidates:
            return []
        bm25 = BM25([tokenize(chunk.text) for chunk in candidates])
        return [(candidates[i], score) for i, score in bm25.rank(query)[:limit]]
