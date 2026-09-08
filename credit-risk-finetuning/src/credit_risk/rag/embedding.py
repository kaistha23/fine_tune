"""Embedding interface, with a deterministic placeholder.

The placeholder is a hashed bag-of-words projection. It is NOT a semantic model: it makes
the pipeline runnable and testable offline and nothing more. Swap it for Qwen3-Embedding
(0.6B) or BGE before any accuracy claim, and re-index - vectors from different models are
not comparable.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

from credit_risk.rag.lexical import tokenize

DIMENSIONS = 256


class Embedder(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...


class HashingEmbedder:
    """Deterministic, dependency-free, and clearly not semantic."""

    model_id = "placeholder-hashing-v1"

    def __init__(self, dimensions: int = DIMENSIONS):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
