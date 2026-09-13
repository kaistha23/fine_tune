"""Submit-only feedback: persist diagnostics, development checks and recommendations."""

from credit_risk.review_store import digest
from credit_risk.workbench.contracts import Case
from credit_risk.workbench.evaluation import assess_training_target

CAUSES = {
    "unknown",
    "data",
    "query",
    "calculation",
    "retrieval",
    "prompt",
    "output_schema",
    "model_behaviour",
}
MAX_FEEDBACK_PER_TEMPLATE_FAMILY = 50


def _recommendations(cause, version, interaction, correction, diagnostics):
    """Offer reviewable suggestions tied to the observed failure category.

    These are deterministic hypotheses. They remain inactive until a user saves a
    version and an evaluation demonstrates an improvement.
    """

    proposed = {"prompt": None, "output_schema": None, "training_example": None}
    if cause in {"model_behaviour", "prompt"}:
        suffix = (
            "\nBefore answering, verify the requested entity, as-of date, jurisdiction, "
            "units and numerical values against the supplied facts and evidence. Abstain "
            "when a material input or supporting source is absent."
        )
        proposed["prompt"] = {
            "before": version["prompt"],
            "after": version["prompt"] + suffix,
            "rationale": "Prompt hypothesis based on a model or prompt feedback category; not evaluated.",
        }
    if cause == "output_schema" or "invalid_json_or_schema" in diagnostics:
        proposed["output_schema"] = {
            "before": version["schema"],
            "after": version["schema"],
            "rationale": (
                "The correction failed the current schema. Define a compatible versioned "
                "contract change before changing the schema; no automatic mutation is proposed."
            ),
        }
    if correction is not None:
        proposed["training_example"] = {
            "before": interaction["output"],
            "after": correction,
            "rationale": (
                "A schema-valid correction is only batch eligible when it has independent "
                "testable expectations, is a model-behaviour issue and is outside protected splits."
            ),
        }
    return proposed


def submit(
    store,
    interaction_id,
    submission_id,
    comment,
    correction=None,
    cause="unknown",
    expectations=None,
    semantic_review=None,
):
    if cause not in CAUSES:
        raise ValueError("Unknown feedback cause")
    if not comment.strip() and correction is None:
        raise ValueError("A comment or correction is required")
    interaction = store.get("answer", interaction_id)
    case = Case.model_validate(interaction["case"])
    version = store.get("version", interaction["version_id"])
    diagnostics = []
    valid = False
    original_correction = correction
    testable = bool(expectations or case.expected)
    check_case = (
        case.model_copy(update={"expected": expectations}) if expectations is not None else case
    )
    if correction is not None:
        metrics, diagnostics, parsed = assess_training_target(check_case, correction, version, semantic_review)
        expected_metrics_pass = all(
            value == (0 if name == "confidence_brier_score" else 1)
            for name, value in metrics.items()
            if name not in {"abstention_precision", "extractive_support_heuristic"}
        )
        valid = parsed is not None and metrics.get("json_validity") == 1 and not diagnostics
        if testable:
            valid = valid and expected_metrics_pass
        if valid:
            correction = parsed["answer"]
    # A correction cannot certify itself. Expectations must come from the captured case
    # or be supplied separately and validated against the correction above.
    protected = case.split in ("test", "oot", "validation") or any(
        c["group_id"] == case.group_id and c["split"] in ("test", "oot", "validation")
        for ds in store.list("dataset")
        for c in ds["cases"]
    )
    eligible = valid and testable and not protected and cause == "model_behaviour"
    if correction is None:
        eligibility_status = "submitted"
    elif not valid:
        eligibility_status = "invalid_correction"
    elif protected:
        eligibility_status = "protected_split_or_group"
    elif not testable:
        eligibility_status = "needs_expected_results"
    elif cause == "unknown":
        eligibility_status = "untriaged"
    elif cause != "model_behaviour":
        eligibility_status = "component_feedback"
    else:
        eligibility_status = "eligible_for_batch"
    proposed = _recommendations(cause, version, interaction, correction, diagnostics)
    record = store.add(
        "feedback",
        {
            "interaction_id": interaction_id,
            "semantic_review": semantic_review,
            "comment": comment,
            "correction": correction,
            "submitted_correction": original_correction,
            "cause": cause,
            "requested_cause": cause,
            "diagnostics": diagnostics,
            "correction_valid": valid,
            "eligible_for_training": eligible,
            "eligibility_status": eligibility_status,
            "has_independent_expectations": testable,
            "failure_cluster": digest(
                {
                    "cause": cause,
                    "diagnostics": sorted(diagnostics),
                    "expected_checks": sorted((expectations or case.expected).keys()),
                }
            ),
            "protected_split_or_group": protected,
            "expectations": expectations,
            "recommendations": proposed,
            "snapshot": interaction,
            "regression_status": "ready"
            if valid and testable
            else "needs_expected_results",
        },
        submission_id,
    )
    if record["regression_status"] == "ready":
        checks = expectations or case.expected
        regression = case.model_copy(
            update={
                "case_id": "feedback-" + submission_id,
                "split": "development",
                "expected": checks,
                "target": correction,
                "equivalence_id": None,
                "distinct_from": [],
                "provenance": {
                    **case.provenance,
                    "feedback_id": submission_id,
                    "training_exposed": eligible,
                },
            }
        )
        store.add(
            "regression",
            {
                "case": regression.model_dump(),
                "source_interaction": interaction_id,
                "source_feedback": submission_id,
                "version_id": interaction["version_id"],
                "label": "Development check; not independent accuracy evidence",
            },
            submission_id,
        )
    return record


def batch_records(store, task):
    latest = {r["interaction_id"]: r for r in store.list("feedback")}
    cases = []
    skipped = []
    duplicate_or_capped = []
    seen_examples = set()
    family_counts = {}
    for r in latest.values():
        if r["snapshot"]["case"]["task"] != task:
            continue
        if not r["eligible_for_training"]:
            skipped.append(r["id"])
            continue
        case = Case.model_validate(r["snapshot"]["case"])
        # Re-check against every registered protected group at export time.
        protected = {
            c["group_id"]
            for ds in store.list("dataset")
            for c in ds["cases"]
            if c["split"] in ("validation", "test", "oot")
        }
        if case.group_id in protected:
            skipped.append(r["id"])
            continue
        fingerprint = digest(
            {"context": case.context_hash(), "question": case.question, "target": r["correction"]}
        )
        family = case.provenance.get("template_family") or "unclassified"
        if fingerprint in seen_examples or family_counts.get(family, 0) >= MAX_FEEDBACK_PER_TEMPLATE_FAMILY:
            duplicate_or_capped.append(r["id"])
            continue
        seen_examples.add(fingerprint)
        family_counts[family] = family_counts.get(family, 0) + 1
        cases.append(
            {
                **case.model_dump(),
                "target": r["correction"],
                "split": "train",
                "provenance": {
                    **case.provenance,
                    "feedback_id": r["id"],
                    "semantic_review": r.get("semantic_review"),
                    "source_version": r["snapshot"]["version_id"],
                },
            }
        )
    return {
        "task": task,
        "cases": cases,
        "skipped": skipped,
        "duplicate_or_capped": duplicate_or_capped,
        "template_family_counts": family_counts,
        "hash": digest(cases),
        "recommended_feedback_fraction": 0.2,
        "note": "Phase-2 input fragment; mix with curated anchor examples, then revalidate the versioned dataset.",
    }
