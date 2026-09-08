#!/usr/bin/env python3
"""Score a fine-tuned (or base) model on the held-out credit-rating test split.

Run it twice — once with --adapter-path and once without — to confirm the fine-tune
actually beat the base model rather than just producing plausible text.

    python scripts/eval.py --adapter-path ./adapters
    python scripts/eval.py                      # base-model baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from credit.metrics import compute_metrics, format_report
from credit.schema import parse_rating

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-8bit"


def load_test_records(path: Path, limit: int | None) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[:limit] if limit else records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter-path", default=None, help="Omit to evaluate the base model as a baseline."
    )
    parser.add_argument("--data", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N examples.")
    parser.add_argument("--out", type=Path, default=None, help="Write metrics JSON here.")
    args = parser.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    records = load_test_records(args.data, args.limit)
    label = args.adapter_path or "base model (no adapters)"
    print(f"Model:   {args.model}")
    print(f"Adapter: {label}")
    print(f"Test set: {args.data} ({len(records)} examples)\n")

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    # Greedy decoding: ratings should be reproducible, not sampled.
    sampler = make_sampler(temp=0.0)

    pairs: list[tuple[str, str | None]] = []
    started = time.time()
    for i, record in enumerate(records, 1):
        messages = record["messages"]
        true_rating = parse_rating(messages[-1]["content"])
        prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
        response = generate(
            model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
            sampler=sampler, verbose=False,
        )
        pairs.append((true_rating, parse_rating(response)))
        if i % 10 == 0 or i == len(records):
            rate = i / (time.time() - started)
            print(f"  {i}/{len(records)}  ({rate:.2f} ex/s)", flush=True)

    metrics = compute_metrics(pairs)
    print(format_report(metrics, pairs))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"model": args.model, "adapter": label, **metrics}, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
