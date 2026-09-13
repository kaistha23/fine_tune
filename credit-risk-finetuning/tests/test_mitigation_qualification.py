import json

import pytest

from credit_risk.evaluation.judge import LocalJudge
from credit_risk.evaluation.qualification import (
    CATEGORIES, qualify_judge, qualify_retrieval, reviewed_label, seal, verify_artifact,
)


def reviewed(label):
    return [{"reviewer_id": "test-reviewer-a", "label": label},
            {"reviewer_id": "test-reviewer-b", "label": label}]


def observations():
    # Synthetic plumbing fixtures, not an actual qualified human-labelled benchmark.
    return [{"group_id": f"test-only-{i}", "split": "qualification",
             "claim": f"Test claim {i}", "sources": {"s": "Test source"},
             "question": f"Test query {i}", "passage": "Test passage",
             "category": sorted(CATEGORIES)[i % len(CATEGORIES)],
             "reviews": reviewed("supported" if i < 80 else "unsupported"),
             "prediction": "supported" if i < 80 else "unsupported",
             "cosine": .9 if i < 80 else .1} for i in range(200)]


def test_qualification_recomputes_labels_not_saved_passed_flag():
    artifact = {"observations": observations(), "passed": True}
    assert qualify_judge(artifact)["passed"]
    for row in artifact["observations"]:
        row["prediction"] = "uncertain"
    assert not qualify_judge(artifact)["passed"]


def test_missing_reviews_and_overlap_are_rejected():
    artifact = {"observations": observations()}
    artifact["observations"][0]["reviews"] = []
    with pytest.raises(ValueError, match="independent"):
        qualify_judge(artifact)
    artifact["observations"] = observations()
    artifact["observations"][1]["group_id"] = artifact["observations"][0]["group_id"]
    with pytest.raises(ValueError, match="independent"):
        qualify_judge(artifact)


def test_disagreement_needs_third_reviewer():
    row = {"reviews": reviewed("supported")}
    row["reviews"][1]["label"] = "unsupported"
    with pytest.raises(ValueError, match="adjudication"):
        reviewed_label(row)
    row["adjudication"] = {"reviewer_id": "test-adjudicator", "label": "unsupported"}
    assert reviewed_label(row) == "unsupported"


def test_retrieval_qualifies_both_positive_and_negative_populations():
    artifact = {"threshold": .72, "observations": observations()}
    assert qualify_retrieval(artifact)["passed"]
    artifact["threshold"] = 1.
    assert not qualify_retrieval(artifact)["passed"]


def test_sealed_artifact_detects_changed_observations():
    artifact = seal({"observations": observations()})
    verify_artifact(artifact)
    artifact["observations"][0]["prediction"] = "unsupported"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_artifact(artifact)


@pytest.mark.parametrize("url,model,candidate,revision", [
    ("https://external.example/v1", "judge", "candidate", "rev"),
    ("http://localhost/v1", "same", "same", "rev"),
    ("http://localhost/v1", "judge", "candidate", None),
])
def test_judge_requires_local_distinct_pinned_identity(url, model, candidate, revision):
    with pytest.raises(ValueError):
        LocalJudge(url, model, candidate, model_revision=revision)


def test_blinded_judge_payload_and_invalid_citation(monkeypatch):
    captured = {}
    class Client:
        def __init__(self, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kw):
            captured.update(kw["json"])
            class Response:
                def raise_for_status(self):
                    pass
                def json(self):
                    return {"choices": [{"message": {"content": json.dumps({
                        "label": "supported", "source_ids": ["unknown"], "reason": "test"})}}]}
            return Response()
    monkeypatch.setattr("httpx.Client", Client)
    judge = LocalJudge("http://localhost/v1", "judge", "candidate-secret", model_revision="revision")
    assert judge.judge("Test claim", {"source": "Test text"}).label == "uncertain"
    assert "candidate-secret" not in json.dumps(captured)
    assert captured["temperature"] == 0


def eligible_scores():
    from credit_risk.evaluation.judge import DECODING, RUBRIC
    from credit_risk.review_store import digest
    identity = {"model": "test-judge", "revision": "test-judge-revision", "rubric_hash": digest(RUBRIC),
                "decoding": DECODING, "version": "semantic-judge-v1"}
    calibration = [{**observations()[0], "group_id": "test-calibration", "split": "calibration",
                    "claim": "Distinct calibration claim",
                    "sources": {"calibration-source": "Distinct calibration source"},
                    "question": "Distinct calibration question",
                    "passage": "Distinct calibration passage"}]
    judge = seal({"observations": observations(), "identity": identity,
                  "calibration_observations": calibration})
    retrieval = seal({"observations": observations(), "threshold": .72,
                      "embedding_signature": "test-embedding-signature", "embedding_revision": "test-embedding-rev",
                      "calibration_observations": calibration})
    return {"qualification": {"judge": judge, "retrieval": retrieval,
                              "training_groups": ["test-training"],
                              "calibration_groups": ["test-calibration"],
                              "exclusions": seal({"training_groups": ["test-training"],
                                                  "calibration_groups": ["test-calibration"]})},
            "candidate_identity": {"model": "test-candidate", "revision": "test-candidate-rev"},
            "judge_identity": identity, "embedding_signature": "test-embedding-signature",
            "retrieval_threshold": .72, "benchmark_manifest": "test-manifest",
            "benchmark_review": seal({"reviewer_id": "test-reviewer", "benchmark_manifest": "test-manifest"}),
            "release_groups": ["test-release"], "group_lineage_complete": True}


def test_qualification_eligibility_requires_matching_reviewed_artifacts():
    from credit_risk.evaluation.qualification import qualification_eligibility
    scores = eligible_scores()
    assert qualification_eligibility(scores) == (True, [])
    scores["release_groups"] = ["test-training"]
    assert not qualification_eligibility(scores)[0]
    scores = eligible_scores()
    scores["retrieval_threshold"] = .5
    assert not qualification_eligibility(scores)[0]
    scores = eligible_scores()
    scores["benchmark_manifest"] = "changed"
    assert not qualification_eligibility(scores)[0]
    scores = eligible_scores()
    scores["candidate_identity"]["revision"] = "test-judge-revision"
    assert not qualification_eligibility(scores)[0]
    scores = eligible_scores()
    scores["qualification"]["judge"]["passed"] = True
    assert not qualification_eligibility(scores)[0]


def test_calibration_content_cannot_reappear_with_a_new_group_id():
    rows = observations()
    duplicate = {**rows[0], "group_id": "different-calibration-group",
                 "split": "calibration"}
    with pytest.raises(ValueError, match="content overlap"):
        qualify_judge({"observations": rows, "calibration_observations": [duplicate]})
    with pytest.raises(ValueError, match="content overlap"):
        qualify_retrieval({"threshold": .72, "observations": rows,
                           "calibration_observations": [duplicate]})


def test_judge_checks_structured_conclusions_and_recommendations(monkeypatch):
    from credit_risk.evaluation.judge import Judgment
    from credit_risk.evaluation.cli import release_report
    from credit_risk.evaluation.gates import ReleaseGates
    from credit_risk.schemas import CreditResponse
    from test_evaluation import gold
    judge = LocalJudge("http://localhost/v1", "test-judge", "test-candidate", model_revision="test-judge-rev")
    seen = []
    def check(claim, sources):
        seen.append((claim, sources))
        return Judgment(label="unsupported", source_ids=list(sources), reason="Synthetic test contradiction")
    monkeypatch.setattr(judge, "judge", check)
    output = CreditResponse(answer_status="ANSWERED", executive_summary="Stage 2 on SICR.",
        conclusions=[{"conclusion_type": "staging", "value": 3, "severity": "high",
                      "evidence_ids": ["SAMA-DOC-4#7.2"]}],
        recommendation_detail={"action": "Reclassify", "rationale_evidence_ids": ["SAMA-DOC-4#7.2"]})
    report = release_report([gold()], {"G1": {"output": output.model_dump()}},
                            ReleaseGates("configs/evaluation_thresholds.yaml"), judge=judge)
    assert any("staging" in text for text, _ in seen)
    assert report["overall"].critical_unsupported_claims > 0
    assert not report["gates"]["passed"]
