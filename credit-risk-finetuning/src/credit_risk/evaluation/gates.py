"""Coverage-aware release decisions. Missing evidence is never a passing score."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

MIN_ELIGIBLE = 30
GATE_RULES = {
    name: (name, "max" if name in {"critical_unsupported_claims", "cross_jurisdiction_retrieval"}
           else "min") for name in (
        "schema_validity", "numerical_agreement", "critical_unsupported_claims",
        "cross_jurisdiction_retrieval", "citation_coverage", "extractive_support_rate",
        "abstention_recall", "prompt_injection_block_rate", "driver_recall",
        "answer_status_correctness", "required_evidence_recall", "semantic_support_rate",
    )
}
ZERO_TOLERANCE = ("critical_unsupported_claims", "cross_jurisdiction_retrieval",
                  "numerical_agreement", "answer_status_correctness")
FRACTIONAL = {"citation_coverage", "driver_recall", "required_evidence_recall"}


def wilson_lower(successes, n):
    if not n:
        return 0.0
    z = 1.959963984540054
    p = successes / n
    return (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)


@dataclass
class GateResult:
    name: str
    scope: str
    threshold: float
    observed: float | None
    passed: bool
    zero_tolerance: bool
    status: str
    denominator: int
    lower_bound: float | None = None

    def describe(self):
        return (f"{self.scope}/{self.name}: observed {self.observed}, n={self.denominator}, "
                f"threshold={self.threshold}, lower_bound={self.lower_bound} -> {self.status}")


class ReleaseGates:
    def __init__(self, path):
        self.thresholds = yaml.safe_load(Path(path).read_text())
        unknown = set(self.thresholds) - set(GATE_RULES)
        if unknown:
            raise ValueError(f"Unknown release gates in thresholds file: {sorted(unknown)}")
        if set(self.thresholds) != set(GATE_RULES):
            raise ValueError("All v2 release gates must be configured")
        for name, value in self.thresholds.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid gate threshold: " + name)
            if GATE_RULES[name][1] == "min" and value > 1:
                raise ValueError("Rate threshold must be <= 1")

    def check(self, card, scope):
        results = []
        for name, threshold in self.thresholds.items():
            observed = getattr(card, name)
            n = card.denominators.get(name, 0)
            lower = None
            comparison = GATE_RULES[name][1]
            breached = observed is not None and (
                observed > threshold if comparison == "max" else observed < threshold
            )
            if not n:
                status = "not_applicable"
            elif n < MIN_ELIGIBLE:
                status = "insufficient_evidence_to_gate"
            elif breached:
                status = "failed"
            elif comparison == "max" or threshold == 1:
                status = "passed"
            else:
                observations = card.observations.get(name, [])
                if len(observations) != n:
                    status = "insufficient_evidence_to_gate"
                else:
                    # Hoeffding handles bounded fractional observations without falsely
                    # treating each expected driver/claim as an independent trial.
                    lower = (max(0, observed - math.sqrt(math.log(20)/(2*n)))
                             if name in FRACTIONAL else wilson_lower(sum(observations), n))
                    status = "passed" if lower >= threshold else "insufficient_evidence_to_gate"
            results.append(GateResult(name, scope, float(threshold), observed, status == "passed",
                                      name in ZERO_TOLERANCE, status, n, lower))
        return results


def evaluate_gates(scores, gates):
    results = gates.check(scores["overall"], "overall")
    for kind in ("portfolio", "task"):
        for name, card in scores["by_" + kind].items():
            results.extend(gates.check(card, f"{kind}:{name}"))
    missing = sorted({"retail", "sme", "corporate"} - set(scores["by_portfolio"]))
    # Optional populations may be absent in a slice, but every configured capability
    # needs coverage overall. A caller cannot drop a whole attack population to pass.
    missing_metrics = [r.name for r in results if r.scope == "overall"
                       and r.status == "not_applicable"]
    failures = [r for r in results if r.status == "failed"]
    insufficient = [r for r in results if r.status == "insufficient_evidence_to_gate"]
    passed = not (failures or insufficient or missing or missing_metrics) and scores["overall"].n > 0
    from credit_risk.evaluation.qualification import qualification_eligibility
    qualified, reasons = qualification_eligibility(scores)
    return {
        "report_version": 2, "passed": passed, "promotable": passed and qualified,
        "coverage_missing": missing, "metric_coverage_missing": missing_metrics,
        "status": "failed" if failures else (
            "passed" if passed else "insufficient_evidence_to_gate"),
        "promotion_block": reasons,
        "blocking_failures": [r.describe() for r in failures if r.zero_tolerance],
        "failures": [r.describe() for r in failures],
        "insufficient_evidence": [r.describe() for r in insufficient],
        "results": [asdict(r) for r in results], "checked": len(results),
    }
