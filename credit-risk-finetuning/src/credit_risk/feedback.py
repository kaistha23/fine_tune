from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from credit_risk.schemas import FeedbackRecord


TRAINING_ROOT_CAUSES = {"model_behaviour"}


def load_feedback(path: Path) -> list[FeedbackRecord]:
    records: list[FeedbackRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(FeedbackRecord.model_validate_json(line))
    return records


def build_training_batch(records: list[FeedbackRecord]) -> tuple[list[dict], dict]:
    eligible: list[dict] = []
    root_causes = Counter(record.root_cause for record in records)
    error_labels = Counter(label for record in records for label in record.error_labels)
    for record in records:
        if (
            record.eligible_for_training
            and record.root_cause in TRAINING_ROOT_CAUSES
            and record.corrected_output
        ):
            eligible.append({
                "example_id": f"feedback-{record.interaction_id}",
                "portfolio": record.portfolio.value,
                "task_type": record.task_type,
                "messages": [
                    {"role": "user", "content": f"Case reference: {record.input_case_id}"},
                    {"role": "assistant", "content": record.corrected_output},
                ],
                "source": "validated_feedback",
            })
    report = {
        "input_records": len(records),
        "training_examples": len(eligible),
        "root_causes": dict(root_causes),
        "error_labels": dict(error_labels),
    }
    return eligible, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    examples, report = build_training_batch(load_feedback(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
