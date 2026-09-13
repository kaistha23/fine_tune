from pathlib import Path

import pytest
import yaml

from credit_risk.data_prep.rules import PolicyRuleError, PolicyRuleRegistry, evaluate_rules
from credit_risk.schemas import CreditFactsheet, Evidence, MetricValue

ROOT = Path(__file__).parents[1]


def registry():
    return PolicyRuleRegistry(ROOT / "configs/policy_rules.yaml", "1.0.0")


def sheet(value=88.0, status="valid", unit="pct"):
    return CreditFactsheet(
        case_id="C",
        obligor_id="O",
        portfolio="corporate",
        jurisdiction="SAMA",
        as_of_date="2026-01-01",
        observation_months=1,
        current_position={},
        calculated_metrics={
            "utilisation_pct": {
                "value": value,
                "unit": unit,
                "validation_status": status,
            }
        },
    )


def source(text="Watchlist escalation applies at utilisation of 85% or more."):
    return Evidence(
        evidence_id="SAMA-DOC-4@1.0#7.2",
        jurisdiction="SAMA",
        document_id="SAMA-DOC-4",
        document_version="1.0",
        section="7.2",
        text=text,
        score=0.95,
    )


def test_rule_fires_only_in_its_declared_scope_and_is_evidence_bound():
    result = evaluate_rules(registry(), sheet(), [source()])
    assert len(result) == 1
    assert result[0].status == "fired" and result[0].holds is True
    outside = sheet()
    outside.jurisdiction = "CBUAE"
    assert evaluate_rules(registry(), outside, [source()]) == []


@pytest.mark.parametrize(
    ("metric", "evidence", "reason"),
    [
        (None, [source()], "metric_unavailable"),
        ({"value": 88, "unit": "pct", "validation_status": "invalid"}, [source()], "metric_invalid"),
        ({"value": 88, "unit": "pct", "validation_status": "valid"}, [], "approved_evidence_unavailable"),
        (
            {"value": 88, "unit": "pct", "validation_status": "valid"},
            [source("No numerical threshold is present.")],
            "threshold_not_in_approved_evidence",
        ),
    ],
)
def test_mandatory_rule_reports_unevaluable_inputs(metric, evidence, reason):
    factsheet = sheet()
    if metric is None:
        factsheet.calculated_metrics = {}
    else:
        factsheet.calculated_metrics["utilisation_pct"] = MetricValue.model_validate(metric)
    result = evaluate_rules(registry(), factsheet, evidence)[0]
    assert result.mandatory and result.status == "unevaluable" and result.reason == reason


def test_registry_rejects_duplicate_ids(tmp_path):
    duplicate = yaml.safe_load((ROOT / "configs/policy_rules.yaml").read_text())
    duplicate["rules"].append(dict(duplicate["rules"][0]))
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(duplicate))
    with pytest.raises(PolicyRuleError, match="Duplicate"):
        PolicyRuleRegistry(path)
