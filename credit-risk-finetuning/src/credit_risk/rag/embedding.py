"""Embedding interface, a real Qwen3 embedder, and a deterministic offline placeholder.

Two implementations, and the difference matters:

HashingEmbedder is a hashed bag-of-words projection. It is NOT semantic - it matches only
on shared surface tokens - and exists so the pipeline is runnable and testable in the
offline container. No retrieval accuracy claim may rest on it.

MLXEmbedder runs Qwen3-Embedding locally through MLX. It is the one to measure against.

Vectors from different models are not comparable, and cosine similarity between them is
meaningless rather than merely poor - it produces confident, wrong neighbours. So every
embedder carries a `signature`, the index records the signature that filled it, and a
mismatch is refused rather than silently scored.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from credit_risk.rag.lexical import tokenize

DIMENSIONS = 256

# Qwen3-Embedding is instruction-aware: queries are prefixed with the task, documents are
# not. Embedding both sides identically costs real recall, so the asymmetry is part of the
# interface rather than something a caller has to remember.
DEFAULT_QUERY_INSTRUCTION = (
    "Given a credit risk question, retrieve the regulatory or policy clause that answers it"
)


@runtime_checkable
class Embedder(Protocol):
    dimensions: int

    @property
    def signature(self) -> str:
        """Identifies the model and dimensionality that produced a vector."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a document."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, which may be encoded differently to a document."""
        ...


class HashingEmbedder:
    """Deterministic, dependency-free, and clearly not semantic."""

    model_id = "placeholder-hashing-v1"

    def __init__(self, dimensions: int = DIMENSIONS):
        self.dimensions = dimensions

    @property
    def signature(self) -> str:
        return f"{self.model_id}:{self.dimensions}"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_query(self, text: str) -> list[float]:
        # Nothing to gain from an instruction prefix in a bag-of-words projection: it
        # would only add its own tokens as noise on the query side.
        return self.embed(text)


class MLXEmbedder:
    """Qwen3-Embedding served locally by MLX.

    Pooling is last-token, not mean: Qwen3-Embedding is a causal model trained so that the
    final position attends over the whole input, and mean pooling over a causal stack
    dilutes that with early positions that have seen almost nothing.

    The model is loaded on first use so that constructing the object - which configuration
    code does eagerly - never blocks on several hundred megabytes of weights.
    """

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3-Embedding-0.6B-8bit",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        max_tokens: int = 2048,
    ):
        self.model_id = model_id
        self.query_instruction = query_instruction
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._dimensions: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - depends on platform
            raise RuntimeError(
                "MLXEmbedder needs mlx-lm, which installs only on Apple Silicon. "
                "Install the 'embedding' extra on macOS/arm64, or use HashingEmbedder "
                "for offline tests."
            ) from exc
        self._model, self._tokenizer = load(self.model_id)
        self._dimensions = int(self._model.args.hidden_size)

    @property
    def dimensions(self) -> int:
        self._load()
        assert self._dimensions is not None
        return self._dimensions

    @property
    def signature(self) -> str:
        return f"{self.model_id}:{self.dimensions}"

    def _encode(self, text: str) -> list[float]:
        import mlx.core as mx

        self._load()
        ids = self._tokenizer.encode(text)
        # Qwen3-Embedding is trained with EOS as the pooled position, so the vector is
        # only correct if that token is actually present.
        eos = self._tokenizer.eos_token_id
        if eos is not None and (not ids or ids[-1] != eos):
            ids = ids + [eos]
        if len(ids) > self.max_tokens:
            # Keep the tail: it holds the pooled position and, in a clause, the operative
            # condition. Dropping it would embed the heading and discard the rule.
            ids = ids[-self.max_tokens:]
        hidden = self._model.model(mx.array([ids]))
        # Normalise in float32. The model runs in bfloat16, where sum-of-squares over a
        # thousand dimensions carries about three decimal digits, so normalising in the
        # model dtype left vectors off unit length by ~2.5e-4. InMemoryPolicyIndex scores
        # with a plain dot product on the assumption that they are unit vectors.
        pooled = hidden[0, -1, :].astype(mx.float32)
        norm = mx.sqrt(mx.sum(pooled * pooled))
        if float(norm) > 0:
            pooled = pooled / norm
        return [float(v) for v in pooled]

    def embed(self, text: str) -> list[float]:
        return self._encode(text)

    def embed_query(self, text: str) -> list[float]:
        return self._encode(f"Instruct: {self.query_instruction}\nQuery: {text}")


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
