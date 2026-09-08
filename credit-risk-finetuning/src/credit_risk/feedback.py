from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from credit_risk.architecture_policy import ArchitecturePolicy
from credit_risk.schemas import FeedbackRecord
from credit_risk.settings import settings

# Only genuine model-behaviour defects become training examples. Everything else is a bug
# in a component that retraining would not fix, so it is routed, not trained on.
TRAINING_ROOT_CAUSES = {"model_behaviour"}

REMEDIATION_ROUTES = {
    "data": "data_pipeline_backlog",
    "schema": "schema_and_query_backlog",
    "query": "schema_and_query_backlog",
    "calculation": "calculation_code_backlog",
    "retrieval": "rag_index_backlog",
    "guardrail": "guardrail_rules_backlog",
    "model_behaviour": "candidate_training_batch",
}


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
    routed: Counter[str] = Counter()
    sql_reviews: Counter[str] = Counter()

    for record in records:
        routed[REMEDIATION_ROUTES.get(record.root_cause, "unrouted")] += 1
        if record.sql_review_status != "not_reviewed":
            sql_reviews[record.sql_review_status] += 1
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
        # Every non-training record still has an owner. Dropping them silently is what
        # let SQL-review corrections disappear in the original implementation.
        "remediation_routes": dict(routed),
        "sql_reviews": dict(sql_reviews),
        "corrections_awaiting_schema_fix": sum(
            1 for r in records if r.corrected_query_plan is not None
        ),
    }
    return eligible, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    # The worker's role is declared in the policy; enforce it here rather than trusting
    # the deployment to have set it correctly.
    policy = ArchitecturePolicy(settings.architecture_policy, settings.architecture_policy_version)
    policy.require(settings.service_role, "read_feedback")
    policy.require(settings.service_role, "classify_root_cause")
    policy.require(settings.service_role, "write_training_batch")

    examples, report = build_training_batch(load_feedback(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
