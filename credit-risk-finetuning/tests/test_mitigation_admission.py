import hashlib

import pytest
from fastapi import HTTPException

from credit_risk.auth import authenticate
from credit_risk.dataset import build_sft_record
from credit_risk.guardrails import is_admissible_training_target, validate_output
from credit_risk.review_store import digest
from credit_risk.schemas import CreditResponse, Evidence
from credit_risk.settings import settings


def test_reviewed_paraphrase_admission_is_separate_from_serving():
    source = Evidence(evidence_id="E", document_id="D", document_version="1", section="1",
                      jurisdiction="SAMA", text="The borrower has missed a payment.", score=.9)
    answer = CreditResponse(answer_status="ANSWERED", executive_summary="A payment was missed by the borrower.",
                            facts=[{"statement": "A payment was missed by the borrower.", "evidence_ids": ["E"]}])
    sheet = {"case_id": "C"}
    assert not validate_output(answer, [source], "C", sheet).passed
    assert not is_admissible_training_target(answer, [source], sheet).passed
    review = {"status": "approved", "reviewer_id": "test-reviewer", "semantic_supported": True,
              "numerics_verified": True, "content_hash": digest({
                  "target": answer.model_dump(mode="json"), "factsheet": sheet,
                  "evidence": [source.model_dump(mode="json")]})}
    assert is_admissible_training_target(answer, [source], sheet, review).passed
    answer.facts[0].evidence_ids = ["invented"]
    assert not is_admissible_training_target(answer, [source], sheet, review).passed


def test_missing_group_rejected_without_obligor_fallback():
    with pytest.raises(ValueError, match="group_id"):
        build_sft_record({"obligor_id": "O"}, '{}', "ews")


@pytest.mark.parametrize("role", ["credit_analyst", "senior_credit_officer", "regulator_liaison"])
def test_supported_roles_authenticate(monkeypatch, role):
    monkeypatch.setattr(settings, "reviewers", {"test-id": {
        "role": role, "token_sha256": hashlib.sha256(b"test-token").hexdigest()}})
    assert authenticate("Bearer test-token") == {"id": "test-id", "role": role}


@pytest.mark.parametrize("header", ["", "Basic test", "Bearer", "Bearer wrong"])
def test_malformed_or_unknown_credentials_rejected(header):
    with pytest.raises(HTTPException) as caught:
        authenticate(header)
    assert caught.value.status_code == 401


def test_unauthorised_role_rejected(monkeypatch):
    monkeypatch.setattr(settings, "reviewers", {"test-id": {
        "role": "intern", "token_sha256": hashlib.sha256(b"test-token").hexdigest()}})
    with pytest.raises(HTTPException) as caught:
        authenticate("Bearer test-token")
    assert caught.value.status_code == 403


def test_real_factsheet_without_group_is_explicitly_rejected_from_feedback():
    from test_factsheet import rows, plan
    from test_feedback import FACTSHEET
    from credit_risk.factsheet import build_factsheet
    from credit_risk.feedback import build_training_batch
    from credit_risk.schemas import FeedbackRecord
    real_sheet = build_factsheet(rows(), plan()).model_dump(mode="json")
    assert "group_id" not in real_sheet
    response = CreditResponse(answer_status="INSUFFICIENT_EVIDENCE", executive_summary="")
    record = FeedbackRecord(interaction_id="test-real-sheet", model_id="test", adapter_version="test",
                            dataset_version="test", portfolio=FACTSHEET["portfolio"], task_type="ews",
                            input_case_id=real_sheet["case_id"], input_factsheet=real_sheet,
                            original_output="wrong", corrected_output=response.model_dump_json(),
                            error_labels=["unsupported_claim"], root_cause="model_behaviour",
                            eligible_for_training=True, reviewer_id="test-reviewer",
                            review_status="approved", quality_score=5)
    examples, report = build_training_batch([record])
    assert examples == [] and report["rejected_missing_group_id"] == 1
