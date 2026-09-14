"""Hybrid retrieval: dense + lexical, fused, then re-checked.

Order matters and is the whole point:

1. Choose the collection from the jurisdiction. A SAMA query never opens CBUAE.
2. Build the access predicate and pass it into both searches, so excluded documents are
   never scored.
3. Fuse the two ranked lists with reciprocal rank fusion.
4. Re-run guardrails.validate_retrieval on what came back, as defence in depth. It should
   have nothing left to find; if it does, that is a bug in the filter, not a save.
"""

from __future__ import annotations

from pathlib import Path

from credit_risk.guardrails import validate_retrieval
from credit_risk.rag.filters import AccessPredicate, RetrievalPolicy, chunk_is_visible
from credit_risk.rag.index import InMemoryPolicyIndex, PolicyIndex
from credit_risk.rag.schemas import AccessContext, PolicyChunk
from credit_risk.schemas import Evidence


class RetrievalError(RuntimeError):
    pass


def reciprocal_rank_fusion(
    arms: list[list[tuple[PolicyChunk, float]]], k: int
) -> list[tuple[PolicyChunk, float]]:
    """Combine ranked lists without needing the arms' scores to be comparable.

    BM25 scores and cosine similarities live on different scales, so fusing on rank rather
    than score is what makes a hybrid honest.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, PolicyChunk] = {}
    for arm in arms:
        for rank, (chunk, _) in enumerate(arm, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk.chunk_id] = chunk
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [(chunks[chunk_id], score) for chunk_id, score in ordered]


def to_evidence(chunk: PolicyChunk, score: float) -> Evidence:
    return Evidence(
        evidence_id=chunk.evidence_id,
        jurisdiction=chunk.jurisdiction,
        document_id=chunk.document_id,
        document_version=chunk.document_version,
        section=chunk.section_id,
        text=chunk.text,
        score=min(max(score, 0.0), 1.0),
        effective_from=chunk.effective_from,
        effective_to=chunk.effective_to,
    )


class PolicyRetriever:
    def __init__(self, policy: RetrievalPolicy, index: PolicyIndex | None = None):
        self.policy = policy
        self.index = index or InMemoryPolicyIndex()

    def predicate_for(self, context: AccessContext) -> AccessPredicate:
        return self.policy.build_predicate(context)

    def retrieve(self, question: str, context: AccessContext) -> dict:
        settings = self.policy.settings
        predicate = self.predicate_for(context)

        dense = self.index.search_dense(question, predicate, int(settings["candidate_k"]))
        lexical = self.index.search_lexical(question, predicate, int(settings["candidate_k"]))
        fused = reciprocal_rank_fusion([dense, lexical], int(settings["rrf_k"]))

        top_k = int(settings["top_k"])
        # Both arms are candidate generators; every candidate uses the same cosine gate.
        chunks = [chunk for chunk, _ in fused]
        if any(not chunk_is_visible(chunk, predicate) for chunk in chunks):
            raise RetrievalError("Pre-search access filter did not hold")
        dense_scores = self.index.score_candidates(question, chunks) if chunks else {}
        budget = int(settings["max_context_chars"])
        evidence = []
        for chunk, _ in fused:
            score = dense_scores[chunk.chunk_id]
            if score < float(settings["min_score"]) or len(chunk.text) > budget:
                continue
            evidence.append(to_evidence(chunk, score))
            budget -= len(chunk.text)
            if len(evidence) == top_k:
                break

        guardrail = validate_retrieval(
            evidence,
            context.jurisdiction,
            float(settings["min_score"]),
            context.as_of_date,
        )
        # The pre-search filter should make cross-jurisdiction and effective-date failures
        # impossible. If one appears here the filter is broken, so fail loudly.
        hard = [
            f
            for f in guardrail.failures
            if f.startswith(("cross_jurisdiction", "not_yet_effective", "superseded"))
        ]
        if hard:
            raise RetrievalError(f"Pre-search access filter did not hold: {sorted(hard)}")

        sufficient = bool(evidence) and not guardrail.failures
        return {
            "question": question,
            "jurisdiction": context.jurisdiction.value,
            "collection": predicate.collection,
            "access_filter": predicate.describe,
            "retrieved": len(evidence),
            "relevance_calibrated": False,
            "score_kind": "cosine_candidate_similarity",
            "evidence": evidence,
            "answer_status": "ANSWERED" if sufficient else "INSUFFICIENT_EVIDENCE",
            "guardrail_failures": guardrail.failures,
            "guardrail_warnings": guardrail.warnings,
        }


def load_retriever(path: str | Path, index: PolicyIndex | None = None) -> PolicyRetriever:
    return PolicyRetriever(RetrievalPolicy(path), index)
