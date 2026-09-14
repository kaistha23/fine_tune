"""Recompute qualification from reviewed observations; never trust a saved passed flag."""
from __future__ import annotations

from credit_risk.review_store import digest

CATEGORIES = {"supported_paraphrase", "misleading_extract", "contradiction", "wrong_citation",
              "numerical_error", "unjustified_abstention"}


def reviewed_label(row):
    reviews = row.get("reviews", [])
    if len(reviews) != 2 or len({r.get("reviewer_id") for r in reviews}) != 2:
        raise ValueError("Two independent named reviews required")
    if any(not r.get("reviewer_id") or r.get("label") not in {"supported", "unsupported"}
           for r in reviews):
        raise ValueError("Invalid human review")
    if reviews[0]["label"] == reviews[1]["label"]:
        return reviews[0]["label"]
    adjudication = row.get("adjudication", {})
    if (not adjudication.get("reviewer_id") or adjudication["reviewer_id"] in
        {r["reviewer_id"] for r in reviews} or
        adjudication.get("label") not in {"supported", "unsupported"}):
        raise ValueError("Disagreement requires independent adjudication")
    return adjudication["label"]


def qualify_judge(artifact):
    from credit_risk.evaluation.gates import wilson_lower
    rows = artifact["observations"]
    if not CATEGORIES <= {r["category"] for r in rows}:
        raise ValueError("Missing judge challenge categories")
    groups = set()
    seen_content = set()
    calibration_content = {
        digest({"claim": row.get("claim"), "sources": row.get("sources")})
        for row in artifact.get("calibration_observations", [])
    }
    if len(calibration_content) != len(artifact.get("calibration_observations", [])):
        raise ValueError("Duplicate judge calibration content")
    populations = {"supported": [], "unsupported": []}
    for row in rows:
        if row.get("split") != "qualification" or not row.get("group_id"):
            raise ValueError("Held-out qualification group required")
        if row["group_id"] in groups:
            raise ValueError("Qualification observations must have independent groups")
        groups.add(row["group_id"])
        if not row.get("claim") or not row.get("sources"):
            raise ValueError("Qualification claim and frozen sources required")
        key = digest({"claim": row["claim"], "sources": row["sources"]})
        if key in seen_content:
            raise ValueError("Duplicate qualification content")
        if key in calibration_content:
            raise ValueError("Judge calibration and qualification content overlap")
        seen_content.add(key)
        label = reviewed_label(row)
        populations[label].append(row.get("prediction") == label)
    bounds = {k: wilson_lower(sum(v), len(v)) for k, v in populations.items()}
    return {"passed": bounds["unsupported"] >= .95 and bounds["supported"] >= .90,
            "lower_bounds": bounds, "groups": sorted(groups)}


def qualify_retrieval(artifact):
    import math
    from credit_risk.evaluation.gates import wilson_lower
    threshold = artifact["threshold"]
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid cosine threshold")
    groups = set()
    seen_content = set()
    calibration_content = {
        digest({"question": row.get("question"), "passage": row.get("passage")})
        for row in artifact.get("calibration_observations", [])
    }
    if len(calibration_content) != len(artifact.get("calibration_observations", [])):
        raise ValueError("Duplicate retrieval calibration content")
    populations = {"supported": [], "unsupported": []}
    for row in artifact["observations"]:
        if row.get("split") != "qualification" or not row.get("group_id"):
            raise ValueError("Held-out qualification group required")
        if row["group_id"] in groups:
            raise ValueError("Retrieval qualification groups must be independent")
        groups.add(row["group_id"])
        if not row.get("question") or not row.get("passage") or not math.isfinite(row["cosine"]):
            raise ValueError("Frozen query, passage and finite cosine required")
        key = digest({"question": row["question"], "passage": row["passage"]})
        if key in seen_content:
            raise ValueError("Duplicate qualification content")
        if key in calibration_content:
            raise ValueError("Retrieval calibration and qualification content overlap")
        seen_content.add(key)
        label = reviewed_label(row)
        accepted = row["cosine"] >= threshold
        populations[label].append(accepted if label == "supported" else not accepted)
    bounds = {k: wilson_lower(sum(v), len(v)) for k, v in populations.items()}
    return {"passed": bounds["supported"] >= .90 and bounds["unsupported"] >= .95,
            "lower_bounds": bounds, "groups": sorted(groups)}



def artifact_calibration_groups(artifact):
    rows = artifact["calibration_observations"]
    if not rows or any(r.get("split") != "calibration" or not r.get("group_id") for r in rows):
        raise ValueError("Frozen reviewed calibration partition required")
    for row in rows:
        reviewed_label(row)
    return {r["group_id"] for r in rows}


def seal(payload):
    return {**payload, "content_hash": digest(payload)}


def verify_artifact(artifact):
    payload = {k: v for k, v in artifact.items() if k != "content_hash"}
    if artifact.get("content_hash") != digest(payload):
        raise ValueError("Qualification artifact hash mismatch")
    return payload


def qualification_eligibility(scores):
    reasons = []
    try:
        bundle = scores["qualification"]
        judge = verify_artifact(bundle["judge"])
        retrieval = verify_artifact(bundle["retrieval"])
        j, r = qualify_judge(judge), qualify_retrieval(retrieval)
        from credit_risk.evaluation.judge import RUBRIC, DECODING
        identity = judge["identity"]
        if (identity["rubric_hash"] != digest(RUBRIC) or identity["decoding"] != DECODING
                or identity["version"] != "semantic-judge-v1" or not identity["revision"]):
            reasons.append("Judge rubric, revision or decoding differs from supported evaluator")
        candidate = scores["candidate_identity"]
        if (not candidate.get("model") or not candidate.get("revision")
                or candidate["model"] == identity["model"] or candidate["revision"] == identity["revision"]):
            reasons.append("Judge must differ from the pinned candidate")
        actual_calibration = artifact_calibration_groups(judge) | artifact_calibration_groups(retrieval)
        if not j["passed"] or not r["passed"]:
            reasons.append("Judge or retrieval qualification below required confidence bound")
        if judge["identity"] != scores["judge_identity"]:
            reasons.append("Judge identity differs from qualification")
        if not retrieval.get("embedding_revision"):
            reasons.append("Immutable embedding revision required")
        if retrieval["embedding_signature"] != scores["embedding_signature"] or (
            retrieval["threshold"] != scores["retrieval_threshold"]
        ):
            reasons.append("Retrieval configuration differs from qualification")
        benchmark = verify_artifact(scores["benchmark_review"])
        if not benchmark.get("reviewer_id") or benchmark["benchmark_manifest"] != scores["benchmark_manifest"]:
            reasons.append("Reviewed benchmark manifest required")
        release = set(scores["release_groups"])
        training = set(bundle["training_groups"])
        calibration = set(bundle["calibration_groups"])
        qualification = set(j["groups"]) | set(r["groups"])
        if actual_calibration != calibration:
            reasons.append("Recorded calibration groups differ from qualification artifacts")
        if not release or not scores.get("group_lineage_complete"):
            reasons.append("Release group lineage incomplete")
        sets = [release, training, calibration, qualification]
        if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
            reasons.append("Training, calibration, qualification and release groups overlap")
        # Even an empty training set must be an explicitly frozen exclusion manifest.
        exclusions = verify_artifact(bundle["exclusions"])
        if set(exclusions["training_groups"]) != training or set(exclusions["calibration_groups"]) != calibration:
            reasons.append("Exclusion manifest mismatch")
    except (KeyError, TypeError, ValueError):
        reasons.append("Reviewed judge, retrieval, benchmark and exclusion artifacts required")
    return not reasons, reasons
