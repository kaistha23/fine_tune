from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from credit_risk.architecture_policy import ArchitecturePolicy
from credit_risk.dataset import DEFAULT_QUESTION
from credit_risk.prompts import PROMPT_VERSION, build_messages
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
    if path.suffix in (".sqlite3", ".db"):
        from credit_risk.feedback_store import FeedbackStore

        return FeedbackStore(path).read_all()
    records: list[FeedbackRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(FeedbackRecord.model_validate_json(line))
    return records


def build_training_batch(records: list[FeedbackRecord]) -> tuple[list[dict], dict]:
    """Route every record, and turn the trainable ones into examples.

    Deduped by interaction here rather than refused at the API. Two reviewers correcting
    the same answer is legitimate and worth keeping for audit, but both records carried
    the same example_id, so the batch silently weighted that interaction twice. The later
    correction wins: it was written with sight of the earlier one.
    """
    by_interaction: dict[str, FeedbackRecord] = {}
    for record in records:
        by_interaction[record.interaction_id] = record
    superseded = len(records) - len(by_interaction)

    eligible: list[dict] = []
    root_causes: Counter[str]
    error_labels: Counter[str]
    routed: Counter[str] = Counter()
    sql_reviews: Counter[str] = Counter()

    root_causes = Counter(r.root_cause for r in by_interaction.values())
    error_labels = Counter(label for r in by_interaction.values() for label in r.error_labels)

    unreconstructable = 0
    for record in by_interaction.values():
        routed[REMEDIATION_ROUTES.get(record.root_cause, "unrouted")] += 1
        if record.sql_review_status != "not_reviewed":
            sql_reviews[record.sql_review_status] += 1
        if not (
            record.eligible_for_training
            and record.root_cause in TRAINING_ROOT_CAUSES
            and record.corrected_output
            and record.reviewer_id
            and record.review_status == "approved"
            and record.quality_score >= 4
        ):
            continue

        # A record that did not capture what the model was shown cannot become a training
        # example. It is skipped and counted, not emitted with a placeholder: the previous
        # behaviour put "Case reference: FB-123" in the user turn, which trains the model
        # to produce a full assessment from an identifier - teaching it to invent.
        if record.input_factsheet is None:
            unreconstructable += 1
            continue

        from credit_risk.guardrails import validate_output
        from credit_risk.schemas import CreditResponse, Evidence

        checked = validate_output(
            CreditResponse.model_validate_json(record.corrected_output),
            [Evidence.model_validate(e) for e in record.input_evidence],
            record.input_case_id,
            record.input_factsheet,
        )
        if not checked.passed:
            continue
        eligible.append(
            {
                "example_id": f"feedback-{record.interaction_id}",
                # Carried so the batch can be split on the same obligor hash as the seed
                # dataset. Written to the provenance sidecar, never into the training line.
                "obligor_id": str(record.input_factsheet.get("obligor_id", record.input_case_id)),
                "portfolio": record.portfolio.value,
                "task_type": record.task_type,
                "messages": [
                    *build_messages(
                        question=record.input_question or DEFAULT_QUESTION,
                        factsheet=record.input_factsheet,
                        evidence=record.input_evidence,
                    ),
                    {"role": "assistant", "content": record.corrected_output},
                ],
                "source": "validated_feedback",
                "prompt_version": PROMPT_VERSION,
            }
        )

    report = {
        "input_records": len(records),
        # Same interaction corrected more than once. Kept in the log, counted once here.
        "superseded_by_later_correction": superseded,
        "training_examples": len(eligible),
        # Surfaced rather than silent: these are corrections a reviewer took the trouble to
        # write that cannot be trained on, which is a defect in what the API records at
        # feedback time, not in the correction.
        "skipped_no_captured_input": unreconstructable,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write train/valid/test.jsonl and their provenance sidecars, "
        "in the same layout as credit_risk.dataset so the two can be merged",
    )
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--out-of-time-from", required=True)
    args = parser.parse_args()

    # The worker's role is declared in the policy; enforce it here rather than trusting
    # the deployment to have set it correctly.
    policy = ArchitecturePolicy(settings.architecture_policy, settings.architecture_policy_version)
    policy.require(settings.service_role, "read_feedback")
    policy.require(settings.service_role, "classify_root_cause")
    policy.require(settings.service_role, "write_training_batch")

    from datetime import date

    from credit_risk.dataset import build_dataset

    records = load_feedback(args.input)
    examples, report = build_training_batch(records)
    eligible_ids = {e["example_id"].removeprefix("feedback-") for e in examples}
    payloads = [
        {
            "case": r.input_factsheet,
            "target": r.corrected_output,
            "task_type": r.task_type,
            "question": r.input_question,
            "evidence": r.input_evidence,
            "review": {
                "reviewer_id": r.reviewer_id,
                "status": r.review_status,
                "quality_score": r.quality_score,
            },
            "data_classification": r.data_classification,
        }
        for r in {r.interaction_id: r for r in records}.values()
        if r.interaction_id in eligible_ids
    ]
    manifest = build_dataset(
        payloads,
        args.output_dir,
        date.fromisoformat(args.out_of_time_from),
        json.loads(args.exclusions.read_text()),
    )
    report["splits"] = manifest["counts"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
