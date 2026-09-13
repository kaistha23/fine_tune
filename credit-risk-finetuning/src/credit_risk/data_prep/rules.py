"""Versioned, hand-authored credit policy rules evaluated before generation."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from credit_risk.data_prep.derivation import OPERATORS, threshold_is_cited
from credit_risk.query_guard import SchemaRegistry
from credit_risk.schemas import CreditFactsheet, Evidence, Jurisdiction, Portfolio


class PolicyRuleError(ValueError):
    pass


class PolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    operator: Literal["gt", "gte", "lt", "lte", "eq"]
    threshold: float
    unit: str = Field(min_length=1)
    action: str = Field(min_length=1)
    mandatory: bool
    jurisdiction: Jurisdiction
    portfolios: list[Portfolio] = Field(min_length=1)
    effective_from: date
    effective_to: date | None = None
    evidence_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_dates_and_numbers(self):
        if not math.isfinite(self.threshold):
            raise ValueError("Rule threshold must be finite")
        if self.effective_to and self.effective_to < self.effective_from:
            raise ValueError("Rule effective_to precedes effective_from")
        return self


class RuleEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: str
    metric: str
    observed: float | None
    operator: str
    threshold: float
    unit: str
    action: str
    evidence_id: str
    mandatory: bool
    status: Literal["fired", "not_fired", "unevaluable"]
    holds: bool | None
    reason: str | None = None


class PolicyRuleRegistry:
    def __init__(
        self,
        path: str | Path,
        expected_version: str | None = None,
        schema_registry: SchemaRegistry | None = None,
    ):
        self.path = Path(path)
        self.data = yaml.safe_load(self.path.read_text())
        if self.data.get("default_deny") is not True:
            raise PolicyRuleError("Policy rule registry must be default-deny")
        if expected_version and str(self.data.get("version")) != expected_version:
            raise PolicyRuleError("Policy rule registry version mismatch")
        try:
            self.rules = [PolicyRule.model_validate(item) for item in self.data.get("rules", [])]
        except ValueError as exc:
            raise PolicyRuleError("Invalid policy rule registry") from exc
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise PolicyRuleError("Duplicate policy rule ID")
        if not self.rules:
            raise PolicyRuleError("Policy rule registry declares no rules")
        if schema_registry:
            for rule in self.rules:
                metric = schema_registry.data.get("metrics", {}).get(rule.metric)
                if not metric or metric.get("unit") != rule.unit:
                    raise PolicyRuleError(f"Rule metric/unit is not governed: {rule.rule_id}")
        action = self.data.get("action_control", {})
        required = {
            "reviewer_id",
            "evidence_id",
            "analysis_tiers",
            "prohibited_actions",
            "high_tier_markers",
            "release",
        }
        if not required <= set(action):
            raise PolicyRuleError("Action control lacks governance or tier configuration")

    @property
    def version(self):
        return str(self.data["version"])

    @property
    def action_control(self):
        return self.data["action_control"]

    def applicable(self, jurisdiction: Jurisdiction, portfolio: Portfolio, as_of: date):
        return [
            rule
            for rule in self.rules
            if rule.jurisdiction == jurisdiction
            and portfolio in rule.portfolios
            and rule.effective_from <= as_of
            and (rule.effective_to is None or rule.effective_to >= as_of)
        ]


def evaluate_rules(
    registry: PolicyRuleRegistry,
    factsheet: CreditFactsheet,
    evidence: list[Evidence],
) -> list[RuleEvaluation]:
    results = []
    for rule in registry.applicable(
        factsheet.jurisdiction, factsheet.portfolio, factsheet.as_of_date
    ):
        metric = factsheet.calculated_metrics.get(rule.metric)
        reason = None
        observed = None
        source = next((item for item in evidence if item.evidence_id == rule.evidence_id), None)
        if source is None:
            reason = "approved_evidence_unavailable"
        elif not threshold_is_cited(rule.threshold, source.text):
            reason = "threshold_not_in_approved_evidence"
        elif metric is None:
            reason = "metric_unavailable"
        elif metric.validation_status != "valid":
            reason = "metric_invalid"
        elif metric.unit != rule.unit:
            reason = "metric_unit_mismatch"
        elif isinstance(metric.value, bool) or not isinstance(metric.value, (int, float)):
            reason = "metric_non_numeric"
        elif not math.isfinite(float(metric.value)):
            reason = "metric_nonfinite"
        else:
            observed = float(metric.value)
        holds = None if reason else OPERATORS[rule.operator](observed, rule.threshold)
        results.append(
            RuleEvaluation(
                rule_id=rule.rule_id,
                metric=rule.metric,
                observed=observed,
                operator=rule.operator,
                threshold=rule.threshold,
                unit=rule.unit,
                action=rule.action,
                evidence_id=rule.evidence_id,
                mandatory=rule.mandatory,
                status="unevaluable" if reason else "fired" if holds else "not_fired",
                holds=holds,
                reason=reason,
            )
        )
    return results


def rule_thresholds(evaluations: list[RuleEvaluation | dict]) -> dict[str, dict[str, Any]]:
    parsed = [
        item if isinstance(item, RuleEvaluation) else RuleEvaluation.model_validate(item)
        for item in evaluations
    ]
    return {item.rule_id: item.model_dump(mode="json") for item in parsed}
