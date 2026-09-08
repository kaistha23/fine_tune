"""Scoring, release gates and champion-challenger comparison (finding C3).

configs/evaluation_thresholds.yaml declared eight release gates and was read by no code, so
no adapter could be promoted on evidence. This package reads them and decides.

Scoring is per portfolio and per task, never one aggregate: both plans warn that an
aggregate hides a material failure in one portfolio.
"""

from credit_risk.evaluation.gates import GateResult, ReleaseGates, evaluate_gates
from credit_risk.evaluation.metrics import ScoreCard, score_cases
from credit_risk.evaluation.runner import compare_adapters

__all__ = [
    "GateResult", "ReleaseGates", "ScoreCard",
    "compare_adapters", "evaluate_gates", "score_cases",
]
