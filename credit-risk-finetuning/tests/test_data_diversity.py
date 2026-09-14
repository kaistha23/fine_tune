import pytest

from credit_risk.data_prep.coverage import coverage_report
from credit_risk.data_prep.diversity import clone_skeleton, validate_diversity
from credit_risk.data_prep.taxonomy import dimensions


def record(identity=1, **updates):
    value = {
        "case_id": f"case-{identity}",
        "group_id": f"group-{identity}",
        "task_type": "claim_verification",
        "situation": "base",
        "portfolio": "corporate",
        "jurisdiction": "SAMA",
        "split": "train",
        "question": f"Is utilisation {identity}% above the threshold?",
        "target": {"supported": True, "observed": identity},
        "provenance": {"template_family": "utilisation-train"},
    }
    value.update(updates)
    return value


def test_unknown_taxonomy_value_is_rejected():
    with pytest.raises(ValueError):
        dimensions(record(situation="ordinary"))


def test_numeric_variants_share_a_clone_skeleton_and_caps_are_enforced():
    assert clone_skeleton(record(12)) == clone_skeleton(record(87))
    with pytest.raises(ValueError, match="clone_skeleton_cap"):
        validate_diversity([record(n) for n in range(51)])


def test_template_family_cannot_cross_splits():
    with pytest.raises(ValueError, match="split_leakage"):
        validate_diversity([record(), record(2, split="test")])


def test_trigger_family_requires_a_near_miss():
    base = record(requires_near_miss=True)
    with pytest.raises(ValueError, match="missing_near_miss"):
        validate_diversity([base])
    near = record(2, situation="near_miss", requires_near_miss=True)
    assert validate_diversity([base, near])["passed"]


def test_coverage_reports_missing_required_cells():
    targets = {
        "minimum_per_cell": 2,
        "required_cells": [
            {
                "task_type": "claim_verification",
                "portfolio": "corporate",
                "jurisdiction": "SAMA",
                "situation": "base",
            }
        ],
    }
    incomplete = coverage_report([record()], targets)
    assert not incomplete["passed"] and incomplete["missing_cells"][0]["actual"] == 1
    assert coverage_report([record(), record(2)], targets)["passed"]
