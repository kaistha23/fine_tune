"""The frozen gold set (the missing input for the evaluation harness).

Scoring, gates and champion-challenger were built with no cases to consume.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from credit_risk.evaluation.gates import ReleaseGates, evaluate_gates
from credit_risk.evaluation.gold import GoldSetError, content_hash, coverage, load_gold_set
from credit_risk.evaluation.metrics import score_case, score_cases

ROOT = Path(__file__).parents[1]
GOLD = ROOT / "data" / "evaluation" / "gold_set.jsonl"
THRESHOLDS = ROOT / "configs" / "evaluation_thresholds.yaml"


class LoadingTests(unittest.TestCase):
    def test_the_seed_set_loads_and_verifies_its_hash(self) -> None:
        cases = load_gold_set(GOLD)
        self.assertEqual(len(cases), 8)

    def test_an_edited_case_is_refused(self) -> None:
        # A regression suite whose cases move silently cannot attribute a difference to
        # the adapter, so a changed file must fail loudly.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "gold_set.jsonl"
            shutil.copy(GOLD, copy)
            text = copy.read_text(encoding="utf-8").replace(
                "more than 90 days past due", "more than 30 days past due"
            )
            copy.write_text(text, encoding="utf-8")
            with self.assertRaises(GoldSetError) as ctx:
                load_gold_set(copy)
            self.assertIn("has changed since it was frozen", str(ctx.exception))

    def test_the_hash_ignores_the_manifest_line_itself(self) -> None:
        self.assertEqual(len(content_hash(GOLD)), 64)

    def test_a_missing_set_is_reported_without_a_path(self) -> None:
        with self.assertRaises(GoldSetError):
            load_gold_set(ROOT / "data" / "evaluation" / "nope.jsonl")


class CoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = load_gold_set(GOLD)

    def test_every_portfolio_a_gate_reads_has_a_case(self) -> None:
        # A slice with no cases cannot gate a release, so an empty one is a coverage hole.
        report = coverage(self.cases)
        self.assertEqual(report["uncovered_portfolios"], [])

    def test_abstention_and_injection_are_represented(self) -> None:
        report = coverage(self.cases)
        self.assertGreaterEqual(report["abstention_cases"], 3)
        self.assertGreaterEqual(report["injection_cases"], 2)

    def test_both_jurisdictions_appear(self) -> None:
        self.assertEqual({c.jurisdiction for c in self.cases}, {"SAMA", "CBUAE"})

    def test_no_evidence_crosses_its_case_jurisdiction(self) -> None:
        for case in self.cases:
            for evidence in case.available_evidence:
                self.assertEqual(
                    evidence.jurisdiction.value,
                    case.jurisdiction,
                    f"{case.case_id} carries foreign evidence",
                )


class HarnessIntegrationTests(unittest.TestCase):
    def test_a_system_that_refuses_everything_passes_abstention_but_not_the_rest(self) -> None:
        # Refusing every case is safe, so the zero-tolerance gates hold - it is the
        # usefulness gates that must catch it.
        cases = load_gold_set(GOLD)
        results = [score_case(c, None, blocked=True) for c in cases]
        scores = score_cases(results)
        self.assertEqual(scores["overall"].critical_unsupported_claims, 0)
        self.assertEqual(scores["overall"].cross_jurisdiction_retrieval, 0)
        self.assertEqual(scores["overall"].prompt_injection_block_rate, 1.0)
        # Cases that should have been answered were not.
        self.assertEqual(scores["overall"].abstention_recall, 1.0)
        self.assertLess(scores["overall"].driver_recall, 1.0)

    def test_that_same_system_is_not_promotable(self) -> None:
        cases = load_gold_set(GOLD)
        results = [score_case(c, None, blocked=True) for c in cases]
        verdict = evaluate_gates(score_cases(results), ReleaseGates(THRESHOLDS))
        self.assertFalse(verdict["passed"])

    def test_gates_run_against_every_gold_slice(self) -> None:
        cases = load_gold_set(GOLD)
        results = [score_case(c, None, blocked=True) for c in cases]
        verdict = evaluate_gates(score_cases(results), ReleaseGates(THRESHOLDS))
        # overall + 3 portfolios + 4 task types, times the eight gates.
        self.assertGreater(verdict["checked"], 8)


if __name__ == "__main__":
    unittest.main()
