"""Risk tiering and action control (Plan section 5, dimension 4).

The model may recommend or draft. It must never execute a material credit decision, and
some requests it must not answer at all.
"""

from __future__ import annotations

from typing import Literal

from credit_risk.schemas import AnswerStatus, CreditResponse

Tier = Literal["low", "medium", "high", "prohibited"]

def _default_control():
    from credit_risk.data_prep.rules import PolicyRuleRegistry
    from credit_risk.settings import settings

    return PolicyRuleRegistry(
        settings.policy_rules, settings.policy_rules_version
    ).action_control


def classify(response: CreditResponse, analysis_type: str, action_control=None) -> tuple[Tier, list[str]]:
    """Return the tier this output must be released under, and why."""
    reasons: list[str] = []
    action_control = action_control or _default_control()
    tier: Tier = action_control["analysis_tiers"].get(analysis_type, "medium")

    body = " ".join(
        [
            response.executive_summary,
            response.recommendation,
            *response.risk_drivers,
            *response.mitigants,
            *response.missing_information,
            *[claim.statement for claim in response.facts],
            *[claim.statement for claim in response.inferences],
        ]
    ).lower()

    for phrase in action_control["prohibited_actions"]:
        if phrase in body:
            reasons.append(f"prohibited_action:{phrase.replace(' ', '_')}")
            tier = "prohibited"
    if tier != "prohibited":
        for phrase in action_control["high_tier_markers"]:
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


def gate(response: CreditResponse, analysis_type: str, action_control=None) -> dict:
    action_control = action_control or _default_control()
    tier, reasons = classify(response, analysis_type, action_control)
    return {
        "risk_tier": tier,
        "release": action_control["release"][tier],
        "reasons": reasons,
        # A recommendation always needs a named human, whatever the tier.
        "human_approval_required": tier != "low" or response.human_approval_required,
        "released": tier == "low" and not response.human_approval_required,
    }
