"""Release gates read from configs/evaluation_thresholds.yaml.

The file declared eight gates and was read by nothing, so promotion had no evidential
basis. Gates are applied to the overall scorecard *and* to every portfolio and task
slice: an aggregate that passes while one portfolio has collapsed is not a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from credit_risk.evaluation.metrics import ScoreCard

# Gate name -> (scorecard attribute, comparison). "min" means at least, "max" at most.
GATE_RULES: dict[str, tuple[str, str]] = {
    "schema_validity": ("schema_validity", "min"),
    "numerical_agreement": ("numerical_agreement", "min"),
    "critical_unsupported_claims": ("critical_unsupported_claims", "max"),
    "cross_jurisdiction_retrieval": ("cross_jurisdiction_retrieval", "max"),
    "citation_coverage": ("citation_coverage", "min"),
    "faithfulness": ("faithfulness", "min"),
    "abstention_recall": ("abstention_recall", "min"),
    "prompt_injection_block_rate": ("prompt_injection_block_rate", "min"),
}

# Gates where any breach blocks promotion outright, per both plans' zero-tolerance list.
ZERO_TOLERANCE = (
    "critical_unsupported_claims",
    "cross_jurisdiction_retrieval",
    "numerical_agreement",
)


@dataclass
class GateResult:
    name: str
    scope: str
    threshold: float
    observed: float
    passed: bool
    zero_tolerance: bool

    def describe(self) -> str:
        direction = "max" if GATE_RULES[self.name][1] == "max" else "min"
        return (
            f"{self.scope}/{self.name}: observed {self.observed} "
            f"vs {direction} {self.threshold} -> {'pass' if self.passed else 'FAIL'}"
        )


class ReleaseGates:
    def __init__(self, path: str | Path):
        with Path(path).open("r", encoding="utf-8") as handle:
            self.thresholds: dict[str, Any] = yaml.safe_load(handle)
        unknown = set(self.thresholds) - set(GATE_RULES)
        if unknown:
            raise ValueError(f"Unknown release gates in thresholds file: {sorted(unknown)}")

    def check(self, card: ScoreCard, scope: str) -> list[GateResult]:
        results: list[GateResult] = []
        for name, threshold in self.thresholds.items():
            attribute, comparison = GATE_RULES[name]
            observed = getattr(card, attribute)
            passed = observed <= threshold if comparison == "max" else observed >= threshold
            results.append(
                GateResult(
                    name=name,
                    scope=scope,
                    threshold=float(threshold),
                    observed=float(observed),
                    passed=bool(passed),
                    zero_tolerance=name in ZERO_TOLERANCE,
                )
            )
        return results


def evaluate_gates(scores: dict[str, Any], gates: ReleaseGates) -> dict[str, Any]:
    """Apply gates overall and to every slice, and say plainly whether this may ship."""
    results: list[GateResult] = list(gates.check(scores["overall"], "overall"))
    for portfolio, card in scores["by_portfolio"].items():
        results.extend(gates.check(card, f"portfolio:{portfolio}"))
    for task, card in scores["by_task"].items():
        results.extend(gates.check(card, f"task:{task}"))

    failures = [r for r in results if not r.passed]
    missing = sorted({"retail", "sme", "corporate"} - set(scores["by_portfolio"]))
    coverage_ok = scores["overall"].n > 0 and not missing
    blocking = [r for r in failures if r.zero_tolerance]
    return {
        "passed": not failures and coverage_ok,
        "promotable": False,
        "coverage_missing": missing,
        "promotion_block": "Requires independent grounding and retrieval calibration plus reviewed benchmark manifest",
        "blocking_failures": [r.describe() for r in blocking],
        "failures": [r.describe() for r in failures],
        "checked": len(results),
    }
