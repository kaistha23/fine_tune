"""Deterministically verify structured numeric claims against governed inputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from credit_risk.schemas import Derivation, Evidence, MetricValue

NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?")
OPERATORS = {
    "gt": lambda observed, threshold: observed > threshold,
    "gte": lambda observed, threshold: observed >= threshold,
    "lt": lambda observed, threshold: observed < threshold,
    "lte": lambda observed, threshold: observed <= threshold,
    "eq": lambda observed, threshold: math.isclose(observed, threshold),
}


@dataclass
class DerivationCheck:
    passed: bool
    failures: list[str] = field(default_factory=list)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def threshold_is_cited(threshold: float, text: str) -> bool:
    for token in NUMBER.findall(text):
        try:
            if math.isclose(float(token.replace(",", "")), threshold):
                return True
        except ValueError:
            continue
    return False


def check_derivation(
    derivation: Derivation,
    factsheet: dict[str, Any],
    evidence: list[Evidence],
    rule_thresholds: dict[str, dict[str, Any]] | None = None,
) -> DerivationCheck:
    failures: list[str] = []
    raw_metric = factsheet.get("calculated_metrics", {}).get(derivation.metric)
    try:
        metric = MetricValue.model_validate(raw_metric) if raw_metric is not None else None
    except ValueError:
        metric = None
    if metric is None:
        failures.append("metric_not_found")
    elif metric.validation_status != "valid":
        failures.append("metric_not_valid")
    else:
        actual = _number(metric.value)
        if actual is None or not math.isclose(actual, derivation.observed):
            failures.append("observed_value_mismatch")
        if metric.unit != derivation.unit:
            failures.append("unit_mismatch")

    source = next(
        (item for item in evidence if item.evidence_id == derivation.threshold_evidence_id), None
    )
    if source is None:
        failures.append("threshold_evidence_not_found")
    if derivation.rule_id:
        rule = (rule_thresholds or {}).get(derivation.rule_id)
        if rule is None:
            failures.append("rule_not_found")
        else:
            if rule.get("metric") != derivation.metric:
                failures.append("rule_metric_mismatch")
            if rule.get("unit") != derivation.unit:
                failures.append("rule_unit_mismatch")
            if rule.get("operator") != derivation.operator:
                failures.append("rule_operator_mismatch")
            if rule.get("evidence_id") != derivation.threshold_evidence_id:
                failures.append("rule_evidence_mismatch")
            rule_threshold = _number(rule.get("threshold"))
            if rule_threshold is None or not math.isclose(rule_threshold, derivation.threshold):
                failures.append("rule_threshold_mismatch")
    elif source is not None and not threshold_is_cited(derivation.threshold, source.text):
        failures.append("threshold_not_in_evidence")

    observed = _number(derivation.observed)
    threshold = _number(derivation.threshold)
    if observed is None or threshold is None:
        failures.append("nonfinite_comparison")
    elif OPERATORS[derivation.operator](observed, threshold) != derivation.holds:
        failures.append("comparison_result_mismatch")
    return DerivationCheck(not failures, sorted(set(failures)))


def check_claim_derivation(claim, factsheet, evidence, rule_thresholds=None) -> DerivationCheck:
    if claim.derivation is None:
        return DerivationCheck(False, ["derivation_missing"])
    result = check_derivation(claim.derivation, factsheet, evidence, rule_thresholds)
    if claim.derivation.threshold_evidence_id not in claim.evidence_ids:
        result.failures.append("threshold_evidence_not_cited_by_claim")
    result.failures = sorted(set(result.failures))
    result.passed = not result.failures
    return result
