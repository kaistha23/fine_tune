"""Build the SFT dataset from validated cases (finding H1).

Two bugs in the original: it wrote a bare JSON array per line, which mlx-lm's chat loader
does not accept - the format is an object with a "messages" key - and it discarded every
provenance field it had just constructed, so example_id, portfolio, task_type,
jurisdiction and dataset_version never reached disk.

Provenance is written to a sidecar rather than into the training line, because mlx-lm
reads each line as a training record and unknown keys are not guaranteed to be ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a credit-risk advisory copilot. Separate facts, model outputs, inference and "
    "recommendations. Do not invent evidence, thresholds or customer facts. Identify missing "
    "information and abstain when evidence is insufficient."
)

DATASET_VERSION = "v1.0.0"
SPLITS = ("train", "valid", "test")


def stable_split(obligor_id: str) -> str:
    """Split by obligor, never by row, so one borrower cannot span train and test."""
    bucket = int(hashlib.sha256(obligor_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "valid"
    return "test"


def build_sft_record(case: dict[str, Any], target: str, task_type: str) -> dict[str, Any]:
    obligor_id = str(case["obligor_id"])
    return {
        "example_id": f"SFT-{hashlib.sha256((obligor_id + task_type).encode()).hexdigest()[:12]}",
        "obligor_id": obligor_id,
        "portfolio": case["portfolio"],
        "task_type": task_type,
        "jurisdiction": case["jurisdiction"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(case, ensure_ascii=False)},
            {"role": "assistant", "content": target},
        ],
        "split": stable_split(obligor_id),
        "dataset_version": DATASET_VERSION,
    }


def to_chat_line(record: dict[str, Any]) -> str:
    """One mlx-lm chat training line: an object keyed by "messages"."""
    return json.dumps({"messages": record["messages"]}, ensure_ascii=False)


def to_provenance_line(record: dict[str, Any]) -> str:
    return json.dumps(
        {key: value for key, value in record.items() if key != "messages"},
        ensure_ascii=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL with case, target and task_type fields")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chat = {s: (args.output_dir / f"{s}.jsonl").open("w", encoding="utf-8") for s in SPLITS}
    meta = {
        s: (args.output_dir / f"{s}.provenance.jsonl").open("w", encoding="utf-8")
        for s in SPLITS
    }
    counts = dict.fromkeys(SPLITS, 0)
    try:
        with args.input.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                payload = json.loads(line)
                record = build_sft_record(
                    payload["case"], payload["target"], payload["task_type"])
                split = record["split"]
                chat[split].write(to_chat_line(record) + "\n")
                meta[split].write(to_provenance_line(record) + "\n")
                counts[split] += 1
    finally:
        for handle in list(chat.values()) + list(meta.values()):
            handle.close()

    for split in SPLITS:
        print(f"{args.output_dir / f'{split}.jsonl'}: {counts[split]} examples")


if __name__ == "__main__":
    main()
