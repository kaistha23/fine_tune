from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from credit_risk.schemas import CreditResponse, Evidence, Jurisdiction, QueryPlan

INJECTION_PATTERNS = [
    r"ignore (all|any|the) previous instructions",
    r"reveal (the )?(system|developer) prompt",
    r"bypass (the )?(guardrails|policy|controls)",
    r"execute unrestricted sql",
]


@dataclass
class GuardrailResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_input(user_text: str, plan: QueryPlan) -> GuardrailResult:
    failures: list[str] = []
    lowered = user_text.lower()
    if any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS):
        failures.append("prompt_injection_detected")
    if plan.date_to > datetime.now(UTC).date():
        failures.append("future_observation_date_not_allowed")
    return GuardrailResult(passed=not failures, failures=failures)


def validate_retrieval(
    evidence: list[Evidence], jurisdiction: Jurisdiction, min_score: float, as_of_date: date
) -> GuardrailResult:
    failures: list[str] = []
    warnings: list[str] = []
    if not evidence:
        return GuardrailResult(False, ["insufficient_evidence"])
    for item in evidence:
        if item.jurisdiction != jurisdiction:
            failures.append(f"cross_jurisdiction:{item.evidence_id}")
        if item.score < min_score:
            failures.append(f"low_relevance:{item.evidence_id}")
        if item.effective_from and item.effective_from > as_of_date:
            failures.append(f"not_yet_effective:{item.evidence_id}")
        if item.effective_to and item.effective_to < as_of_date:
            failures.append(f"superseded:{item.evidence_id}")
    return GuardrailResult(not failures, sorted(set(failures)), sorted(set(warnings)))


def validate_output(
    response: CreditResponse,
    evidence: list[Evidence],
    factsheet_case_id: str | None = None,
    factsheet: dict | None = None,
    rule_evaluations: list | None = None,
) -> GuardrailResult:
    """Check every material fact against something a reviewer can open.

    The factsheet is a citable source, not just context. Without it a fact drawn from
    validated data had no way to be stated: citing the factsheet was an unknown_citation
    and citing nothing was a material_fact_without_citation, so a correct answer about an
    obligor's own position was unreleasable however well evidenced. The accepted handle is
    exactly this case's id - not any string beginning with CASE - because the point of the
    check is that a citation resolves to a specific artefact.

    Nothing is weakened by allowing it: every number in the factsheet was computed in
    Python from rows the data service validated, which is a stronger provenance than a
    retrieved passage.
    """
    failures: list[str] = []
    available_ids = {item.evidence_id for item in evidence}
    if factsheet_case_id:
        available_ids.add(factsheet_case_id)
    source_text = {item.evidence_id: item.text for item in evidence}
    if factsheet_case_id and factsheet:
        source_text[factsheet_case_id] = factsheet_statements(factsheet)
    for claim in response.facts:
        cited = " ".join(source_text.get(eid, "") for eid in claim.evidence_ids)
        if not supported_text(claim.statement, cited):
            failures.append("unsupported_claim")
        if not claim.evidence_ids:
            failures.append("material_fact_without_citation")
        for evidence_id in claim.evidence_ids:
            if evidence_id not in available_ids:
                failures.append(f"unknown_citation:{evidence_id}")
        if claim.derivation:
            from credit_risk.data_prep.derivation import check_claim_derivation

            from credit_risk.data_prep.rules import rule_thresholds

            rules = rule_thresholds(rule_evaluations or [])
            derivation = check_claim_derivation(claim, factsheet or {}, evidence, rules)
            failures.extend(f"invalid_derivation:{reason}" for reason in derivation.failures)
            if claim.derivation.rule_id:
                evaluated = rules.get(claim.derivation.rule_id)
                if evaluated and evaluated.get("holds") != claim.derivation.holds:
                    failures.append("rule_contradiction:" + claim.derivation.rule_id)
    structured_ids = [
        evidence_id
        for item in [*response.conclusions, *response.risk_driver_details]
        for evidence_id in item.evidence_ids
    ]
    if response.recommendation_detail:
        structured_ids.extend(response.recommendation_detail.rationale_evidence_ids)
    for evidence_id in structured_ids:
        if evidence_id not in available_ids:
            failures.append(f"unknown_citation:{evidence_id}")
    # Until a validated entailment evaluator is configured, only extractive statements
    # are automatically verified. Free analytical prose remains a review draft.
    all_sources = " ".join(source_text.values())
    statements = [
        response.executive_summary,
        response.recommendation,
        *response.missing_information,
        *response.risk_drivers,
        *response.mitigants,
        *[c.statement for c in response.inferences],
        *[c.basis for c in response.inferences],
    ]
    for statement in statements:
        if statement and not supported_text(statement, all_sources):
            failures.append("unverified_narrative")
    if not response.human_approval_required and response.recommendation:
        failures.append("credit_recommendation_requires_human_approval")
    return GuardrailResult(not failures, sorted(set(failures)))


def supported_text(statement: str, source: str) -> bool:
    """Conservative extractive verification, not a claim of semantic entailment."""
    def normalise(s):
        return re.sub(r"\s+", " ", s).strip().rstrip(".").casefold()
    text = normalise(statement)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", source)
    return bool(text) and any(text == normalise(sentence) for sentence in sentences)


def factsheet_statements(sheet: dict) -> str:
    """Stable field=value assertions; numbers retain their field and units."""
    statements = []

    def walk(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else key)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            statements.append(f"{path} = {value}.")

    walk(sheet)
    return " ".join(statements)


def is_extractive_copy(response, evidence, factsheet_case_id=None, factsheet=None):
    """Serving acceptance only; no claim of semantic entailment."""
    return validate_output(response, evidence, factsheet_case_id, factsheet)


def is_admissible_training_target(
    response, evidence, factsheet, review=None, rule_evaluations=None
):
    """Permit supported paraphrases only with context-bound semantic/numeric review.

    The normal dataset approval and provenance requirements still apply at the builder.
    A review is bound to the exact target and sources so it cannot be reused after edits.
    """
    from credit_risk.review_store import digest

    check = validate_output(
        response, evidence, factsheet.get("case_id"), factsheet, rule_evaluations
    )
    hard = [f for f in check.failures if f not in {"unsupported_claim", "unverified_narrative"}]
    soft = [f for f in check.failures if f in {"unsupported_claim", "unverified_narrative"}]
    if "unsupported_claim" in soft:
        from credit_risk.data_prep.derivation import check_claim_derivation

        unsupported_without_derivation = any(
            not supported_text(
                claim.statement,
                " ".join(
                    factsheet_statements(factsheet)
                    if evidence_id == factsheet.get("case_id")
                    else next((item.text for item in evidence if item.evidence_id == evidence_id), "")
                    for evidence_id in claim.evidence_ids
                ),
            )
            and not check_claim_derivation(claim, factsheet, evidence).passed
            for claim in response.facts
        )
        if not unsupported_without_derivation:
            soft.remove("unsupported_claim")
    review = review or {}
    bound = digest({"target": response.model_dump(mode="json"), "factsheet": factsheet,
                    "evidence": [e.model_dump(mode="json") for e in evidence]})
    verified = (review.get("reviewer_id") and review.get("status") == "approved"
                and review.get("semantic_supported") is True
                and review.get("numerics_verified") is True
                and review.get("content_hash") == bound)
    if soft and not verified:
        hard.append("semantic_and_numeric_review_required")
    # Unknown/uncited material facts cannot be waived by a semantic review.
    return GuardrailResult(not hard, hard)
