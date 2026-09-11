"""Champion-challenger comparison and the promotion decision.

A candidate is promoted only if it passes every gate, improves the cluster it was trained
to fix, and degrades no portfolio. The champion is never overwritten - both plans are
explicit - so this returns a decision, it does not perform one.
"""

from __future__ import annotations

from typing import Any

from credit_risk.evaluation.gates import ReleaseGates, evaluate_gates
from credit_risk.evaluation.metrics import ScoreCard

# A drop larger than this in any portfolio slice counts as a material regression.
REGRESSION_TOLERANCE = 0.02

COMPARED = (
    "citation_coverage",
    "faithfulness",
    "abstention_recall",
    "numerical_agreement",
    "driver_recall",
    "prompt_injection_block_rate",
)


def _deltas(champion: ScoreCard, candidate: ScoreCard) -> dict[str, float]:
    return {name: round(getattr(candidate, name) - getattr(champion, name), 6) for name in COMPARED}


def compare_adapters(
    champion_scores: dict[str, Any], candidate_scores: dict[str, Any], gates: ReleaseGates
) -> dict[str, Any]:
    candidate_gates = evaluate_gates(candidate_scores, gates)

    regressions: list[str] = []
    if not champion_scores.get("case_set_hash") or champion_scores.get(
        "case_set_hash"
    ) != candidate_scores.get("case_set_hash"):
        regressions.append("Benchmark case sets differ or have no manifest")
    if not champion_scores.get("benchmark_manifest") or champion_scores.get(
        "benchmark_manifest"
    ) != candidate_scores.get("benchmark_manifest"):
        regressions.append(
            "Benchmark inputs, prompts or evidence snapshots differ or are unrecorded"
        )
    for task, champion_card in champion_scores["by_task"].items():
        candidate_card = candidate_scores["by_task"].get(task)
        if candidate_card is None:
            regressions.append(f"task:{task} missing from candidate")
            continue
        for name, delta in _deltas(champion_card, candidate_card).items():
            if delta < -REGRESSION_TOLERANCE:
                regressions.append(f"task:{task}/{name} regressed")
    for portfolio, champion_card in champion_scores["by_portfolio"].items():
        candidate_card = candidate_scores["by_portfolio"].get(portfolio)
        if candidate_card is None:
            regressions.append(f"portfolio:{portfolio} missing from the candidate run")
            continue
        for name, delta in _deltas(champion_card, candidate_card).items():
            if delta < -REGRESSION_TOLERANCE:
                regressions.append(f"portfolio:{portfolio}/{name} fell by {abs(delta):.3f}")

    overall = _deltas(champion_scores["overall"], candidate_scores["overall"])
    improved = [name for name, delta in overall.items() if delta > 0]

    promote = candidate_gates["promotable"] and not regressions and bool(improved)
    return {
        "promote": promote,
        # Stated even when promoting, so the reason is always on the record.
        "decision": "promote_candidate" if promote else "retain_champion",
        "gates": candidate_gates,
        "overall_deltas": overall,
        "improved_metrics": improved,
        "portfolio_regressions": regressions,
        "champion_overwritten": False,
    }
