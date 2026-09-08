"""Risk tiering and action control (Plan section 5, dimension 4).

The model may recommend or draft. It must never execute a material credit decision, and
some requests it must not answer at all.
"""
from __future__ import annotations

from typing import Literal

from credit_risk.schemas import AnswerStatus, CreditResponse

Tier = Literal["low", "medium", "high", "prohibited"]

# Analysis type to the tier its output carries.
ANALYSIS_TIERS: dict[str, Tier] = {
    "factsheet": "low",
    "credit_deterioration": "medium",
    "ews_analysis": "medium",
    "email_draft": "medium",
    "policy_qa": "medium",
}

# Language that indicates the model has crossed from advising into deciding.
PROHIBITED_ACTIONS = (
    "approve the facility", "decline the application", "approved the loan",
    "we hereby approve", "final credit decision", "override the model",
    "waive the covenant", "amend the policy",
)

# Conclusions a human must own even when correctly reasoned.
HIGH_TIER_MARKERS = (
    "rating recommendation", "recommend downgrade", "recommend upgrade",
    "sicr", "stage migration", "reclassify to stage", "limit increase",
    "covenant breach", "write-off",
)

RELEASE = {
    "low": "auto_release",
    "medium": "analyst_review_required",
    "high": "senior_credit_approval_required",
    "prohibited": "blocked",
}


def classify(response: CreditResponse, analysis_type: str) -> tuple[Tier, list[str]]:
    """Return the tier this output must be released under, and why."""
    reasons: list[str] = []
    tier: Tier = ANALYSIS_TIERS.get(analysis_type, "medium")

    body = " ".join([
        response.executive_summary, response.recommendation,
        *[claim.statement for claim in response.facts],
        *[claim.statement for claim in response.inferences],
    ]).lower()

    for phrase in PROHIBITED_ACTIONS:
        if phrase in body:
            reasons.append(f"prohibited_action:{phrase.replace(' ', '_')}")
            tier = "prohibited"
    if tier != "prohibited":
        for phrase in HIGH_TIER_MARKERS:
            if phrase in body:
                reasons.append(f"material_conclusion:{phrase.replace(' ', '_')}")
                tier = "high"
        if response.recommendation and tier == "low":
            reasons.append("carries_a_recommendation")
            tier = "medium"
        if response.answer_status == AnswerStatus.ESCALATE:
            reasons.append("model_requested_escalation")
            tier = "high"
    return tier, sorted(set(reasons))


def gate(response: CreditResponse, analysis_type: str) -> dict:
    tier, reasons = classify(response, analysis_type)
    return {
        "risk_tier": tier,
        "release": RELEASE[tier],
        "reasons": reasons,
        # A recommendation always needs a named human, whatever the tier.
        "human_approval_required": tier != "low" or response.human_approval_required,
        "released": tier == "low",
    }
