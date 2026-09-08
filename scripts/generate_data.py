#!/usr/bin/env python3
"""Generate the synthetic credit-rating dataset for mlx-lm LoRA fine-tuning.

Writes train.jsonl / valid.jsonl / test.jsonl in mlx-lm chat format.

    python scripts/generate_data.py --n 3000 --out ./data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from credit.rationale import compose_rationale
from credit.scorecard import rate_issuer, sample_issuer
from credit.schema import RATINGS, to_chat_example

# Natural sampling produces almost no AAA or CCC issuers, which starves exactly the
# classes the model is worst at. Rejection-sample toward an even spread, then give up
# gracefully rather than looping forever on buckets the scorecard rarely reaches.
_ATTEMPT_BUDGET_MULTIPLIER = 60


def build_examples(n: int, rng: np.random.Generator, balance: bool) -> list[tuple[str, dict]]:
    per_bucket_target = n // len(RATINGS) if balance else n
    counts: Counter[str] = Counter()
    examples: list[tuple[str, dict]] = []
    attempts = 0
    budget = n * _ATTEMPT_BUDGET_MULTIPLIER

    while len(examples) < n and attempts < budget:
        attempts += 1
        # Sweeping stress across its full range is what populates the AAA and CCC tails;
        # sector distributions on their own bunch everything into A through BB.
        issuer = sample_issuer(rng, stress=float(rng.uniform(-1.2, 1.2)))
        rating, factors = rate_issuer(issuer, rng)
        if balance and counts[rating] >= per_bucket_target:
            continue
        rationale = compose_rationale(issuer, rating, factors, rng)
        counts[rating] += 1
        examples.append((rating, to_chat_example(issuer, rating, rationale)))

    # Top up with unfiltered draws if the rare buckets could not be filled.
    while len(examples) < n:
        issuer = sample_issuer(rng, stress=float(rng.uniform(-1.2, 1.2)))
        rating, factors = rate_issuer(issuer, rng)
        rationale = compose_rationale(issuer, rating, factors, rng)
        counts[rating] += 1
        examples.append((rating, to_chat_example(issuer, rating, rationale)))

    return examples


def stratified_split(
    examples: list[tuple[str, dict]], rng: np.random.Generator
) -> dict[str, list[dict]]:
    """Split 80/10/10 within each rating so every split covers the full scale."""
    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    by_rating: dict[str, list[dict]] = {r: [] for r in RATINGS}
    for rating, example in examples:
        by_rating[rating].append(example)

    for rating in RATINGS:
        group = by_rating[rating]
        rng.shuffle(group)
        n = len(group)
        n_valid = max(1, int(round(n * 0.10))) if n >= 10 else 0
        n_test = max(1, int(round(n * 0.10))) if n >= 10 else 0
        splits["valid"].extend(group[:n_valid])
        splits["test"].extend(group[n_valid : n_valid + n_test])
        splits["train"].extend(group[n_valid + n_test :])

    for split in splits.values():
        rng.shuffle(split)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000, help="Total examples to generate.")
    parser.add_argument("--out", type=Path, default=Path("data"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=17, help="PRNG seed.")
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Sample ratings as they naturally fall instead of evening out the buckets.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    examples = build_examples(args.n, rng, balance=not args.no_balance)
    splits = stratified_split(examples, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        path = args.out / f"{name}.jsonl"
        with path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"{path}: {len(records)} examples")

    distribution = Counter(rating for rating, _ in examples)
    print("\nRating distribution:")
    for rating in RATINGS:
        count = distribution[rating]
        bar = "#" * int(50 * count / max(distribution.values()))
        print(f"  {rating:<4} {count:>5}  {bar}")

    print("\nSample record:")
    print(json.dumps(splits["train"][0], indent=2)[:900])


if __name__ == "__main__":
    main()
