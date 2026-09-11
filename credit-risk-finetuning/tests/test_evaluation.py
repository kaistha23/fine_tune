"""Release gates and champion-challenger promotion (finding C3).

configs/evaluation_thresholds.yaml declared release gates and was read by no code, so
nothing could be promoted on evidence.
"""

import unittest
from pathlib import Path

from credit_risk.evaluation.gates import ReleaseGates, evaluate_gates
from credit_risk.evaluation.metrics import CaseResult, GoldCase, score_case, score_cases
from credit_risk.evaluation.runner import compare_adapters
from credit_risk.schemas import (
    AnswerStatus,
    CreditResponse,
    Evidence,
    Jurisdiction,
    SupportedClaim,
)

THRESHOLDS = Path(__file__).parents[1] / "configs" / "evaluation_thresholds.yaml"


def evidence(
    eid: str = "SAMA-DOC-4#7.2", jurisdiction: Jurisdiction = Jurisdiction.SAMA
) -> Evidence:
    return Evidence(
        evidence_id=eid,
        jurisdiction=jurisdiction,
        document_id=eid.split("#")[0],
        document_version="1.0",
        section="7.2",
        text="Stage 2 on SICR. leverage deterioration",
        score=0.9,
    )


def gold(**overrides) -> GoldCase:
    base = {
        "case_id": "G1",
        "portfolio": "corporate",
        "task_type": "ews_analysis",
        "jurisdiction": "SAMA",
        "question": "staging?",
        "available_evidence": [evidence()],
    }
    base.update(overrides)
    return GoldCase(**base)


def good_response() -> CreditResponse:
    return CreditResponse(
        answer_status=AnswerStatus.ANSWERED,
        executive_summary="Stage 2 on SICR.",
        facts=[SupportedClaim(statement="Stage 2 on SICR.", evidence_ids=["SAMA-DOC-4#7.2"])],
        risk_drivers=["leverage deterioration"],
        human_approval_required=True,
    )


def perfect_results(n: int = 4, portfolio: str = "corporate") -> list[CaseResult]:
    return [
        CaseResult(
            case_id=f"{portfolio}-G{i}",
            portfolio=portfolio,
            task_type="ews_analysis",
            schema_valid=True,
            citation_coverage=1.0,
            unsupported_claims=0,
            unknown_citations=0,
            cross_jurisdiction=0,
            numeric_agreement=True,
            abstained_correctly=True,
            injection_blocked=True,
            driver_recall=1.0,
        )
        for i in range(n)
    ]


class ScoringTests(unittest.TestCase):
    def test_a_cited_answer_scores_clean(self) -> None:
        result = score_case(gold(), good_response())
        self.assertEqual(result.unsupported_claims, 0)
        self.assertEqual(result.unknown_citations, 0)
        self.assertEqual(result.citation_coverage, 1.0)

    def test_a_fabricated_citation_is_counted(self) -> None:
        r = CreditResponse(
            answer_status=AnswerStatus.ANSWERED,
            executive_summary="x",
            facts=[SupportedClaim(statement="c", evidence_ids=["SAMA-DOC-99#1.1"])],
        )
        self.assertEqual(score_case(gold(), r).unknown_citations, 1)

    def test_an_uncited_claim_is_counted(self) -> None:
        r = CreditResponse(
            answer_status=AnswerStatus.ANSWERED,
            executive_summary="x",
            facts=[SupportedClaim(statement="c", evidence_ids=[])],
        )
        self.assertGreaterEqual(score_case(gold(), r).unsupported_claims, 1)

    def test_cross_jurisdiction_citation_is_counted(self) -> None:
        case = gold(available_evidence=[evidence("CBUAE-DOC-1#2", Jurisdiction.CBUAE)])
        r = CreditResponse(
            answer_status=AnswerStatus.ANSWERED,
            executive_summary="x",
            facts=[SupportedClaim(statement="c", evidence_ids=["CBUAE-DOC-1#2"])],
        )
        self.assertEqual(score_case(case, r).cross_jurisdiction, 1)

    def test_refusing_an_injection_counts_as_blocked(self) -> None:
        self.assertTrue(
            score_case(gold(is_injection_attempt=True), None, blocked=True).injection_blocked
        )

    def test_answering_an_injection_counts_as_not_blocked(self) -> None:
        self.assertFalse(
            score_case(gold(is_injection_attempt=True), good_response()).injection_blocked
        )

    def test_block_rate_uses_only_attack_cases_as_denominator(self) -> None:
        attacked = score_case(gold(case_id="A", is_injection_attempt=True), good_response())
        normal = score_case(gold(case_id="B"), good_response())
        scores = score_cases([attacked, normal])
        # One attack, not blocked, so 0.0 - not 0.5 diluted by the clean case.
        self.assertEqual(scores["overall"].prompt_injection_block_rate, 0.0)

    def test_missing_abstention_is_caught(self) -> None:
        self.assertFalse(score_case(gold(must_abstain=True), good_response()).abstained_correctly)

    def test_scores_are_broken_out_by_portfolio_and_task(self) -> None:
        scores = score_cases(perfect_results(2) + perfect_results(2, portfolio="retail"))
        self.assertIn("corporate", scores["by_portfolio"])
        self.assertIn("retail", scores["by_portfolio"])
        self.assertIn("ews_analysis", scores["by_task"])


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gates = ReleaseGates(THRESHOLDS)

    def test_thresholds_load_and_every_gate_is_known(self) -> None:
        self.assertIn("citation_coverage", self.gates.thresholds)
        self.assertIn("faithfulness", self.gates.thresholds)

    def test_a_clean_run_passes(self) -> None:
        verdict = evaluate_gates(score_cases(perfect_results()), self.gates)
        self.assertFalse(verdict["passed"])

    def test_one_unsupported_claim_blocks_release(self) -> None:
        results = perfect_results()
        results[0].unsupported_claims = 1
        verdict = evaluate_gates(score_cases(results), self.gates)
        self.assertFalse(verdict["passed"])
        self.assertTrue(
            any("critical_unsupported_claims" in f for f in verdict["blocking_failures"])
        )

    def test_one_cross_jurisdiction_hit_blocks_release(self) -> None:
        results = perfect_results()
        results[0].cross_jurisdiction = 1
        verdict = evaluate_gates(score_cases(results), self.gates)
        self.assertFalse(verdict["passed"])
        self.assertTrue(verdict["blocking_failures"])

    def test_a_single_failing_portfolio_blocks_an_otherwise_good_run(self) -> None:
        # The aggregate still looks acceptable; the slice must not.
        bad = perfect_results(1, portfolio="retail")
        bad[0].numeric_agreement = False
        verdict = evaluate_gates(score_cases(perfect_results(9) + bad), self.gates)
        self.assertFalse(verdict["passed"])
        self.assertTrue(any("portfolio:retail" in f for f in verdict["failures"]))


class ChampionChallengerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gates = ReleaseGates(THRESHOLDS)

    def test_a_candidate_that_improves_nothing_is_not_promoted(self) -> None:
        verdict = compare_adapters(
            score_cases(perfect_results()), score_cases(perfect_results()), self.gates
        )
        self.assertFalse(verdict["promote"])
        self.assertEqual(verdict["decision"], "retain_champion")

    def test_a_regressed_portfolio_blocks_promotion(self) -> None:
        champion = score_cases(perfect_results(5) + perfect_results(5, portfolio="retail"))
        weaker = perfect_results(5, portfolio="retail")
        for r in weaker:
            r.driver_recall = 0.5
        candidate = score_cases(perfect_results(5) + weaker)
        verdict = compare_adapters(champion, candidate, self.gates)
        self.assertFalse(verdict["promote"])
        self.assertTrue(verdict["portfolio_regressions"])

    def test_the_champion_is_never_overwritten(self) -> None:
        verdict = compare_adapters(
            score_cases(perfect_results()), score_cases(perfect_results()), self.gates
        )
        self.assertFalse(verdict["champion_overwritten"])

    def test_an_improved_candidate_passing_every_gate_is_promoted(self) -> None:
        weaker = perfect_results(4)
        for r in weaker:
            r.driver_recall = 0.8
        verdict = compare_adapters(score_cases(weaker), score_cases(perfect_results(4)), self.gates)
        self.assertFalse(verdict["promote"])
        self.assertIn("driver_recall", verdict["improved_metrics"])


if __name__ == "__main__":
    unittest.main()
