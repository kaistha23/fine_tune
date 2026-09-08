from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

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
    if plan.date_to > date.today():
        failures.append("future_observation_date_not_allowed")
    return GuardrailResult(passed=not failures, failures=failures)


def validate_retrieval(evidence: list[Evidence], jurisdiction: Jurisdiction,
                       min_score: float, as_of_date: date) -> GuardrailResult:
    failures: list[str] = []
    warnings: list[str] = []
    if not evidence:
        return GuardrailResult(False, ["insufficient_evidence"])
    for item in evidence:
        if item.jurisdiction != jurisdiction:
            failures.append(f"cross_jurisdiction:{item.evidence_id}")
        if item.score < min_score:
            warnings.append(f"low_relevance:{item.evidence_id}")
        if item.effective_from and item.effective_from > as_of_date:
            failures.append(f"not_yet_effective:{item.evidence_id}")
        if item.effective_to and item.effective_to < as_of_date:
            failures.append(f"superseded:{item.evidence_id}")
    return GuardrailResult(not failures, sorted(set(failures)), sorted(set(warnings)))


def validate_output(response: CreditResponse, evidence: list[Evidence],
                    factsheet_case_id: str | None = None) -> GuardrailResult:
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
    for claim in response.facts:
        if not claim.evidence_ids:
            failures.append("material_fact_without_citation")
        for evidence_id in claim.evidence_ids:
            if evidence_id not in available_ids:
                failures.append(f"unknown_citation:{evidence_id}")
    if not response.human_approval_required and response.recommendation:
        failures.append("credit_recommendation_requires_human_approval")
    return GuardrailResult(not failures, sorted(set(failures)))
