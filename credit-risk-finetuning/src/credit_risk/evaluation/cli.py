"""Frozen release scoring for captured outputs or a local oMLX candidate."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from credit_risk.evaluation.gates import ReleaseGates, evaluate_gates
from credit_risk.evaluation.gold import load_gold_set
from credit_risk.evaluation.metrics import SCORING_VERSION, score_case, score_cases
from credit_risk.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from credit_risk.review_store import digest
from credit_risk.schemas import CreditResponse


def release_report(cases, predictions, thresholds, *, judge=None, metadata=None):
    """Shared by CLI and workbench. Predictions contain output/errors, never gold answers."""
    allowed_metadata = {"qualification", "benchmark_review", "generation", "candidate_identity",
                        "embedding_signature", "retrieval_threshold", "outputs_hash",
                        "prompt_hash", "prompt_version"}
    if set(metadata or {}) - allowed_metadata:
        raise ValueError("Metadata may only contain identity and qualification fields")
    if set(predictions) - {c.case_id for c in cases}:
        raise ValueError("Predictions contain unknown benchmark case IDs")
    if judge and judge.identity["revision"] == (metadata or {}).get("candidate_identity", {}).get("revision"):
        raise ValueError("Judge and candidate revisions must differ")
    results, records = [], []
    for case in cases:
        prediction = predictions.get(case.case_id, {"provider_failed": True})
        output = prediction.get("output")
        failed = prediction.get("provider_failed", False)
        result = score_case(case, output, prediction.get("numerics"),
                            prediction.get("blocked", False), failed)
        judgments = []
        if judge is not None:
            result.semantic_support = False
            if output is not None and result.schema_valid and not failed:
                parsed = (CreditResponse.model_validate_json(output) if isinstance(output, str)
                          else CreditResponse.model_validate(output))
                result.semantic_support, judgments = judge.assess_response(parsed, case)
                result.unsupported_claims += sum(j.label == "unsupported" for j in judgments)
        results.append(result)
        records.append({"result": asdict(result), "prediction": prediction,
                        "judgments": [j.model_dump() for j in judgments]})
    scores = score_cases(results)
    scores.update(metadata or {})
    scores["benchmark_manifest"] = digest({
        "cases": [asdict(c) for c in cases], "prompt_version": PROMPT_VERSION,
        "prompt_hash": scores.get("prompt_hash", digest(SYSTEM_PROMPT)),
        "scoring_version": SCORING_VERSION,
        "thresholds": thresholds.thresholds,
        "generation": scores.get("generation"),
    })
    scores["group_lineage_complete"] = all(c.group_id for c in cases)
    scores["release_groups"] = sorted({c.group_id for c in cases if c.group_id})
    if judge:
        scores["judge_identity"] = judge.identity
    scores["gates"] = evaluate_gates(scores, thresholds)
    scores["per_case"] = records
    scores["report_version"] = 2
    return scores


def serializable(report):
    return json.loads(json.dumps(report, default=lambda v: v.as_dict(), allow_nan=False))



def load_report(path):
    from credit_risk.evaluation.metrics import ScoreCard
    from credit_risk.evaluation.qualification import verify_artifact

    result = verify_artifact(json.loads(Path(path).read_text()))
    if result.get("report_version") != 2 or result.get("scoring_version") != SCORING_VERSION:
        raise ValueError("Only current, versioned reports can be compared; rerun historical evaluations")
    result["overall"] = ScoreCard(**result["overall"])
    for key in ("by_portfolio", "by_task"):
        result[key] = {k: ScoreCard(**v) for k, v in result[key].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--outputs", type=Path, help="JSON mapping case IDs to prediction records")
    source.add_argument("--model", help="Candidate model served by local oMLX")
    parser.add_argument("--candidate-model", help="Model identity for captured outputs")
    parser.add_argument("--model-revision", required=True, help="Immutable candidate checkpoint hash")
    parser.add_argument("--thresholds", type=Path, default=Path("configs/evaluation_thresholds.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-revision")
    parser.add_argument("--champion", type=Path, help="Sealed v2 champion report for comparison")
    parser.add_argument("--target-metric", default="driver_recall")
    parser.add_argument("--metadata", type=Path, help="Qualification, benchmark review and retrieval identities")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output must be a new report path")
    cases = load_gold_set(args.gold)
    from credit_risk.settings import settings
    metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
    candidate_model = args.model or args.candidate_model
    metadata["candidate_identity"] = {"model": candidate_model, "revision": args.model_revision}
    judge = None
    if args.judge_model:
        from credit_risk.evaluation.judge import LocalJudge
        judge = LocalJudge(settings.omlx_base_url, args.judge_model,
                           candidate_model, settings.omlx_api_key,
                           args.judge_revision)
    if args.outputs:
        predictions = json.loads(args.outputs.read_text())
        metadata["outputs_hash"] = digest(predictions)
    else:
        from credit_risk.omlx_client import OMLXClient
        client = OMLXClient(settings.omlx_base_url, args.model, settings.omlx_api_key)
        metadata["generation"] = {"temperature": .1, "top_p": .9, "top_k": 20,
                                  "max_tokens": 2500, "enable_thinking": False}
        predictions = {}
        for case in cases:
            try:
                output = client.generate_credit_response(case.question, case.factsheet,
                                                         case.available_evidence)
                predictions[case.case_id] = {"output": output.model_dump(mode="json")}
            except Exception as exc:
                predictions[case.case_id] = {"provider_failed": True, "error": type(exc).__name__}
    report = release_report(cases, predictions, ReleaseGates(args.thresholds),
                            judge=judge, metadata=metadata)
    if args.champion:
        from credit_risk.evaluation.runner import compare_adapters
        report["comparison"] = compare_adapters(load_report(args.champion), report,
                                                ReleaseGates(args.thresholds), args.target_metric)
        report["target_metric"] = args.target_metric
    from credit_risk.evaluation.qualification import seal
    with args.out.open("x") as handle:
        json.dump(seal(serializable(report)), handle, indent=2, allow_nan=False)
    print(json.dumps({"report": str(args.out), "gates": report["gates"]["status"],
                      "promotable": report["gates"]["promotable"]}))


if __name__ == "__main__":
    main()
