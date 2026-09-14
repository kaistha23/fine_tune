"""Embedding interface, a real Qwen3 embedder, and a deterministic offline placeholder.

Two implementations, and the difference matters:

HashingEmbedder is a hashed bag-of-words projection. It is NOT semantic - it matches only
on shared surface tokens - and exists so the pipeline is runnable and testable in the
offline container. No retrieval accuracy claim may rest on it.

MLXEmbedder runs Qwen3-Embedding locally through MLX, in-process on the Mac.

OMLXEmbedder is what the deployed API uses: the API container has no Metal, so it asks
the native oMLX server for embeddings over the same HTTP route it already uses for chat.

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


class OMLXEmbedder:
    """Embeddings from the native oMLX server over HTTP.

    This is the one the deployed API uses. The API runs in a Linux container with no
    Metal, so it cannot run MLXEmbedder in-process - but oMLX is already running natively
    on the Mac for chat completions and exposes /v1/embeddings, so embeddings take the
    same route as inference rather than needing a second native service.

    Dimensionality is discovered from the server on first use. Hard-coding it would let a
    model change pass unnoticed until the vectors were already in the index.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
                 timeout: float = 30.0, model_revision: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_revision = model_revision
        self.api_key = api_key
        self.query_instruction = query_instruction
        self.timeout = timeout
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._dimensions = len(self._request("dimension probe"))
        return self._dimensions

    @property
    def signature(self) -> str:
        instruction = hashlib.sha256(self.query_instruction.encode()).hexdigest()[:16]
        return f"{self.model}:{self.dimensions}:{self.model_revision or 'unrecorded'}:{instruction}"

    def _request(self, text: str) -> list[float]:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": text},
                headers=headers,
            )
            response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        if (not vector or any(type(value) not in (int, float) or not math.isfinite(value)
                              for value in vector)):
            raise ValueError("Embedding server returned invalid vector")
        if self._dimensions is not None and len(vector) != self._dimensions:
            raise ValueError("Embedding server changed vector dimensions")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Embedding server returned unusable vector norm")
        self._dimensions = len(vector)
        # Servers differ on whether they normalise. InMemoryPolicyIndex scores with a
        # plain dot product, so do it here rather than trusting the server.
        return [value / norm for value in vector] if norm else vector

    def embed(self, text: str) -> list[float]:
        return self._request(text)

    def embed_query(self, text: str) -> list[float]:
        return self._request(f"Instruct: {self.query_instruction}\nQuery: {text}")


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
