"""Okapi BM25, implemented here rather than pulled in as a dependency.

Hybrid retrieval needs a lexical arm: regulatory text turns on exact terms - a clause
number, "significant increase in credit risk", a defined threshold - that dense vectors
blur together. BM25 is forty lines and fully specified, and keeping it in-tree means the
core retrieval path does not depend on an optional extra.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, documents: list[list[str]]):
        self.documents = documents
        self.lengths = [len(doc) for doc in documents]
        self.average_length = (sum(self.lengths) / len(documents)) if documents else 0.0
        self.frequencies = [Counter(doc) for doc in documents]
        document_frequency: Counter[str] = Counter()
        for doc in documents:
            document_frequency.update(set(doc))
        total = len(documents)
        # Standard BM25+ style idf floor, so a term in almost every document cannot go
        # negative and drag a genuinely matching passage below a non-matching one.
        self.idf = {
            term: max(math.log((total - count + 0.5) / (count + 0.5) + 1.0), 1e-9)
            for term, count in document_frequency.items()
        }

    def score(self, query: list[str], index: int) -> float:
        if not self.documents:
            return 0.0
        frequencies = self.frequencies[index]
        length = self.lengths[index] or 1
        total = 0.0
        for term in query:
            if term not in frequencies:
                continue
            frequency = frequencies[term]
            denominator = frequency + K1 * (
                1 - B + B * length / (self.average_length or 1)
            )
            total += self.idf.get(term, 0.0) * frequency * (K1 + 1) / denominator
        return total

    def rank(self, query: str) -> list[tuple[int, float]]:
        terms = tokenize(query)
        scored = [(i, self.score(terms, i)) for i in range(len(self.documents))]
        return sorted(
            [pair for pair in scored if pair[1] > 0.0],
            key=lambda pair: pair[1], reverse=True,
        )
