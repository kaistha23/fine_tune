"""Run local qualification against independently reviewed, frozen labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from credit_risk.evaluation.qualification import (
    qualify_judge, qualify_retrieval, reviewed_label, seal,
)


def select_threshold(rows):
    """Select on calibration only; qualification never influences this threshold."""
    if not rows or any(r["split"] != "calibration" for r in rows):
        raise ValueError("Reviewed calibration observations required")
    labels = [reviewed_label(r) for r in rows]
    if set(labels) != {"supported", "unsupported"}:
        raise ValueError("Both relevant and irrelevant calibration passages required")
    thresholds = sorted({0.0, 1.0, *[max(0, min(1, r["cosine"])) for r in rows]})
    def quality(t):
        recall = sum(r["cosine"] >= t for r, y in zip(rows, labels) if y == "supported") / labels.count("supported")
        rejection = sum(r["cosine"] < t for r, y in zip(rows, labels) if y == "unsupported") / labels.count("unsupported")
        return min(recall, rejection), rejection, recall, t
    return max(thresholds, key=quality)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["judge", "retrieval"])
    parser.add_argument("--input", required=True, type=Path,
                        help="JSONL reviewed calibration/qualification rows")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-revision")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output artifact must be new")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    calibration = [r for r in rows if r.get("split") == "calibration"]
    qualification = [r for r in rows if r.get("split") == "qualification"]
    if not calibration or not qualification or len(calibration) + len(qualification) != len(rows):
        raise ValueError("Separate calibration and qualification partitions required")
    if {r["group_id"] for r in calibration} & {r["group_id"] for r in qualification}:
        raise ValueError("Calibration and qualification group overlap")
    for row in rows:
        reviewed_label(row)
    from credit_risk.settings import settings
    if args.kind == "judge":
        from credit_risk.evaluation.judge import LocalJudge
        judge = LocalJudge(settings.omlx_base_url, args.judge_model, args.candidate_model,
                           settings.omlx_api_key, args.judge_revision)
        for row in rows:
            answer = judge.judge(row["claim"], row["sources"])
            row["prediction"] = answer.label
            row["judgment"] = answer.model_dump()
        artifact = {"version": 1, "identity": judge.identity,
                    "observations": qualification, "calibration_observations": calibration}
        result = qualify_judge(artifact)
    else:
        from credit_risk.rag.embedding import cosine
        from credit_risk.rag.factory import build_embedder
        if settings.offline_test_mode or not settings.embedding_model or not settings.embedding_revision:
            raise ValueError("Semantic embeddings and CR_EMBEDDING_REVISION required for retrieval qualification")
        embedder = build_embedder(settings)
        for row in rows:
            row["cosine"] = cosine(embedder.embed_query(row["question"]),
                                   embedder.embed(row["passage"]))
        artifact = {"version": 1, "embedding_signature": embedder.signature,
                    "embedding_revision": settings.embedding_revision,
                    "threshold": select_threshold(calibration),
                    "observations": qualification, "calibration_observations": calibration}
        result = qualify_retrieval(artifact)
    with args.out.open("x") as handle:
        json.dump(seal(artifact), handle, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
