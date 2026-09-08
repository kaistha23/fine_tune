"""ACL-aware, jurisdiction-isolated retrieval over policy and regulatory documents.

The handover repository ran Qdrant as a container and had no retrieval code at all, so
SAMA/CBUAE separation existed only as a post-hoc equality check in
guardrails.validate_retrieval - applied to evidence that nothing ever retrieved.

Separation here is structural. Each jurisdiction is a separate collection, chosen before a
query is built, and every remaining control is a pre-search filter rather than a
post-search check. Nothing is filtered out after the fact, because by then the wrong
documents have already been read.
"""

from credit_risk.rag.retriever import PolicyRetriever, RetrievalError
from credit_risk.rag.schemas import AccessContext, PolicyChunk

__all__ = ["AccessContext", "PolicyChunk", "PolicyRetriever", "RetrievalError"]
