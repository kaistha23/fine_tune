import json
from dataclasses import replace
from pathlib import Path

import pytest

from credit_risk.evaluation.gates import ReleaseGates, evaluate_gates, wilson_lower
from credit_risk.evaluation.metrics import GoldCase, score_case, score_cases
from credit_risk.evaluation.runner import compare_adapters
from credit_risk.schemas import CreditResponse
from test_evaluation import gold, good_response, perfect_results

GATES = Path("configs/evaluation_thresholds.yaml")


def test_missing_population_is_not_a_zero_score():
    card = score_cases([score_case(gold(), good_response())])["overall"]
    assert card.abstention_recall is None
    assert card.denominators["abstention_recall"] == 0
    rows = ReleaseGates(GATES).check(card, "task:ews")
    assert next(r for r in rows if r.name == "abstention_recall").status == "not_applicable"
    verdict = evaluate_gates(score_cases([score_case(gold(), good_response())]), ReleaseGates(GATES))
    assert "abstention_recall" in verdict["metric_coverage_missing"]
    assert not verdict["passed"]


def test_shipped_gold_cannot_gate_even_with_perfect_observations():
    from credit_risk.evaluation.gold import load_gold_set
    cases = load_gold_set("data/evaluation/gold_set.jsonl")
    results = []
    for case in cases:
        row = perfect_results(1, case.portfolio)[0]
        row.case_id, row.task_type = case.case_id, case.task_type
        row.must_abstain, row.is_injection_attempt = case.must_abstain, case.is_injection_attempt
        row.answer_status_correct = row.grounding_verified = row.semantic_support = True
        row.required_evidence_recall = 1.
        results.append(row)
    report = evaluate_gates(score_cases(results), ReleaseGates(GATES))
    assert report["status"] == "insufficient_evidence_to_gate"
    assert not report["passed"] and not report["promotable"]


def test_small_sample_breach_is_reported_as_insufficient_evidence():
    rows = perfect_results(8)
    rows[0].unsupported_claims = 1
    report = evaluate_gates(score_cases(rows), ReleaseGates(GATES))
    gate = next(result for result in report["results"]
                if result["scope"] == "overall"
                and result["name"] == "critical_unsupported_claims")
    assert gate["status"] == "insufficient_evidence_to_gate"
    assert not report["passed"] and not report["promotable"]


def test_shared_groups_do_not_inflate_independent_sample():
    rows = perfect_results(100)
    for row in rows:
        row.group_id = "same-obligor-group"
    card = score_cases(rows)["overall"]
    assert card.n == 100
    assert card.denominators["schema_validity"] == 1


def test_exact_correctness_uses_coverage_not_impossible_wilson_bound():
    rows = perfect_results(30)
    card = score_cases(rows)["overall"]
    gate = next(r for r in ReleaseGates(GATES).check(card, "overall") if r.name == "schema_validity")
    assert gate.passed and gate.lower_bound is None
    assert wilson_lower(30, 30) < 1


def test_fractional_recall_is_not_treated_as_binomial_successes():
    rows = perfect_results(100)
    for row in rows:
        row.driver_recall = .95
    gate = next(r for r in ReleaseGates(GATES).check(score_cases(rows)["overall"], "overall")
                if r.name == "driver_recall")
    assert gate.lower_bound < .95
    assert gate.lower_bound != pytest.approx(wilson_lower(95, 100))


@pytest.mark.parametrize("output", [None, {"answer_status": "ANSWERED", "executive_summary": ""},
                                    {"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}])
def test_mute_or_incorrect_abstention_cannot_pass(output):
    result = score_case(gold(required_risk_drivers=["leverage deterioration"]), output)
    assert not result.answer_status_correct
    assert result.driver_recall == 0


def test_provider_failure_is_not_a_successful_refusal():
    result = score_case(gold(must_abstain=True), None, blocked=True, provider_failed=True)
    assert not result.schema_valid and not result.abstained_correctly


def test_foreign_retrieval_is_counted_even_when_uncited():
    from test_evaluation import evidence
    from credit_risk.schemas import Jurisdiction

    case = gold(available_evidence=[evidence("CBUAE-DOC-1#2", Jurisdiction.CBUAE)])
    answer = CreditResponse(answer_status="ANSWERED", executive_summary="A local conclusion.")
    assert score_case(case, answer).cross_jurisdiction == 1


def test_factsheet_citation_and_numerical_prediction_are_scored():
    case = GoldCase("g", "corporate", "ews", "SAMA", "Stage?",
                    factsheet_case_id="C", factsheet={"case_id": "C", "current_position": {"stage": 1}},
                    required_evidence_ids=["C"], expected_numerics={"current_position.stage": 1})
    answer = CreditResponse(answer_status="ANSWERED", executive_summary="current_position.stage = 1.",
                            facts=[{"statement": "current_position.stage = 1.", "evidence_ids": ["C"]}])
    result = score_case(case, answer)
    assert result.unknown_citations == 0 and result.grounding_verified
    assert result.numeric_agreement and result.required_evidence_recall == 1
    assert not score_case(case, answer, numerics={"current_position.stage": 2}).numeric_agreement


def test_promotable_flag_does_not_bypass_failed_gates(monkeypatch):
    from credit_risk.evaluation import runner
    champion = score_cases(perfect_results(3))
    candidate = score_cases(perfect_results(3))
    champion["overall"].driver_recall = .8
    champion["benchmark_manifest"] = candidate["benchmark_manifest"] = "same"
    monkeypatch.setattr(runner, "evaluate_gates", lambda *a: {"passed": False, "promotable": True})
    assert not compare_adapters(champion, candidate, ReleaseGates(GATES))["promote"]


def test_report_cli_preserves_failures_and_does_not_overwrite(tmp_path, monkeypatch):
    from credit_risk.evaluation.cli import main
    import sys
    outputs = tmp_path / "outputs.json"
    outputs.write_text("{}")
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--gold", "data/evaluation/gold_set.jsonl",
                                    "--outputs", str(outputs), "--model-revision", "test-checkpoint",
                                    "--out", str(out)])
    main()
    report = json.loads(out.read_text())
    assert len(report["per_case"]) == 8
    assert report["report_version"] == 2 and not report["gates"]["passed"]
    assert all(r["result"]["provider_failed"] for r in report["per_case"])
    with pytest.raises(SystemExit):
        main()


def test_missing_candidate_metric_blocks_comparison():
    left = score_cases(perfect_results(2))
    right = score_cases([replace(r, driver_applicable=False) for r in perfect_results(2)])
    report = compare_adapters(left, right, ReleaseGates(GATES))
    assert report["overall_deltas"]["driver_recall"] is None
    assert any("unavailable" in reason for reason in report["portfolio_regressions"])


def test_missing_output_stays_in_required_evidence_denominator():
    case = gold(required_evidence_ids=["required"])
    card = score_cases([score_case(case, None, provider_failed=True)])["overall"]
    assert card.required_evidence_recall == 0
    assert card.denominators["required_evidence_recall"] == 1


def test_metadata_cannot_override_computed_scores():
    from credit_risk.evaluation.cli import release_report
    with pytest.raises(ValueError, match="Metadata"):
        release_report([], {}, ReleaseGates(GATES), metadata={"overall": {"n": 10000}})
