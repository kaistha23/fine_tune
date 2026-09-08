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


def stable_split(obligor_id: str) -> str:
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
        "portfolio": case["portfolio"],
        "task_type": task_type,
        "jurisdiction": case["jurisdiction"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(case, ensure_ascii=False)},
            {"role": "assistant", "content": target},
        ],
        "split": stable_split(obligor_id),
        "dataset_version": "v1.0.0",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL with case and target fields")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "valid", "test")
    }
    try:
        with args.input.open("r", encoding="utf-8") as source:
            for line in source:
                payload = json.loads(line)
                record = build_sft_record(payload["case"], payload["target"], payload["task_type"])
                handles[record["split"]].write(json.dumps(record["messages"], ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    main()
