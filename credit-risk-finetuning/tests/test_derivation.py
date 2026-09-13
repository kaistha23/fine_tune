import copy

import pytest

from credit_risk.data_prep.derivation import check_claim_derivation
from credit_risk.guardrails import is_admissible_training_target, validate_output
from credit_risk.schemas import CreditResponse, Evidence, SupportedClaim


@pytest.fixture
def evidence():
    return [
        Evidence(
            evidence_id="SAMA-CIRC-4#7.2",
            jurisdiction="SAMA",
            document_id="SAMA-CIRC-4",
            document_version="1",
            section="7.2",
            text="Escalate when utilisation is at least 85.0%.",
            score=0.95,
        )
    ]


@pytest.fixture
def factsheet():
    return {
        "case_id": "CASE-1",
        "calculated_metrics": {
            "utilisation_pct": {
                "value": 88.0,
                "unit": "pct",
                "formula_id": "utilisation_pct_v2",
                "validation_status": "valid",
            }
        },
    }


def claim(**changes):
    derivation = {
        "metric": "utilisation_pct",
        "observed": 88.0,
        "operator": "gte",
        "threshold": 85.0,
        "unit": "pct",
        "threshold_evidence_id": "SAMA-CIRC-4#7.2",
        "holds": True,
    }
    derivation.update(changes)
    return SupportedClaim(
        statement="Utilisation of 88.0% is at least the 85.0% escalation threshold.",
        evidence_ids=["SAMA-CIRC-4#7.2"],
        derivation=derivation,
    )


def test_verified_derivation_admits_numeric_claim_without_semantic_review(factsheet, evidence):
    response = CreditResponse(
        answer_status="ANSWERED", executive_summary="", facts=[claim()]
    )
    assert check_claim_derivation(response.facts[0], factsheet, evidence).passed
    assert is_admissible_training_target(response, evidence, factsheet).passed


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        ({"metric": "missing"}, "metric_not_found"),
        ({"observed": 87.0}, "observed_value_mismatch"),
        ({"unit": "probability"}, "unit_mismatch"),
        ({"threshold": 84.0}, "threshold_not_in_evidence"),
        ({"holds": False}, "comparison_result_mismatch"),
        ({"threshold_evidence_id": "MISSING"}, "threshold_evidence_not_found"),
    ],
)
def test_each_incorrect_derivation_component_is_rejected(
    factsheet, evidence, change, failure
):
    result = check_claim_derivation(claim(**change), factsheet, evidence)
    assert not result.passed
    assert failure in result.failures


def test_invalid_or_nonfinite_metric_cannot_support_a_derivation(factsheet, evidence):
    invalid = copy.deepcopy(factsheet)
    invalid["calculated_metrics"]["utilisation_pct"]["validation_status"] = "invalid"
    assert "metric_not_valid" in check_claim_derivation(claim(), invalid, evidence).failures
    nonfinite = copy.deepcopy(factsheet)
    nonfinite["calculated_metrics"]["utilisation_pct"]["value"] = float("nan")
    assert "observed_value_mismatch" in check_claim_derivation(
        claim(), nonfinite, evidence
    ).failures


def test_threshold_source_must_be_a_claim_citation(factsheet, evidence):
    derived = claim()
    derived.evidence_ids = ["CASE-1"]
    result = check_claim_derivation(derived, factsheet, evidence)
    assert result.failures == ["threshold_evidence_not_cited_by_claim"]


def test_serving_validation_rejects_an_incorrect_derivation(factsheet, evidence):
    response = CreditResponse(
        answer_status="ANSWERED", executive_summary="", facts=[claim(holds=False)]
    )
    result = validate_output(response, evidence, "CASE-1", factsheet)
    assert "invalid_derivation:comparison_result_mismatch" in result.failures
