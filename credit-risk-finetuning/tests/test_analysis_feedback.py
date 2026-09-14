"""The input side of the feedback loop (the gap that made retraining impossible).

TRAINING_ROOT_CAUSES is {"model_behaviour"}, but nothing could produce a record with that
root cause: /v1/query/review only yields schema, query, calculation and data causes, and
/v1/analyse returned no handle to attach a correction to. Every correction an analyst
might have written was unrecordable, so the loop had no input at all.

Most of what follows is about what must *not* reach the training set. A correction that
looks helpful and is wrong is worse than none, because it is indistinguishable from a good
one once it is in the batch.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from http_helpers import TestClient

from credit_risk import api
from credit_risk.feedback import build_training_batch
from credit_risk.feedback_store import FeedbackStore
from credit_risk.interaction_store import InteractionNotFound, InteractionStore
from credit_risk.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from credit_risk.schemas import AnswerStatus, InteractionRecord, Jurisdiction, Portfolio

FACTSHEET = {
    # Synthetic fixture with explicit lineage; real factsheets without it are rejected.
    "group_id": "synthetic-feedback-group",
    "case_id": "CASE-OBL-0008-2026-01-15",
    "obligor_id": "OBL-0008",
    "portfolio": "corporate",
    "jurisdiction": "SAMA",
    "current_position": {"stage": 1, "days_past_due": 0},
}
EVIDENCE = [
    {
        "evidence_id": "SAMA-CIRC-4#7.2",
        "jurisdiction": "SAMA",
        "document_id": "SAMA-CIRC-4",
        "document_version": "2.0",
        "section": "7.2",
        "text": "Stage 2 on a significant increase in credit risk.",
        "score": 0.93,
    }
]


def correction(evidence_ids: list[str] | None = None, **overrides) -> str:
    body = {
        "answer_status": "ANSWERED",
        "executive_summary": "current_position.stage = 1.",
        "facts": [
            {
                "statement": "current_position.days_past_due = 0.",
                "evidence_ids": evidence_ids
                if evidence_ids is not None
                else ["CASE-OBL-0008-2026-01-15"],
            }
        ],
        "human_approval_required": True,
    }
    body.update(overrides)
    return json.dumps(body)


def interaction(interaction_id: str = "IX-1", **overrides) -> InteractionRecord:
    base = {
        "interaction_id": interaction_id,
        "model_id": "qwen3.5-9b",
        "adapter_version": "base",
        "dataset_version": "none",
        "prompt_version": PROMPT_VERSION,
        "portfolio": Portfolio.CORPORATE,
        "jurisdiction": Jurisdiction.SAMA,
        "task_type": "credit_deterioration",
        "case_id": "CASE-OBL-0008-2026-01-15",
        "question": "Has the obligor deteriorated?",
        "factsheet": FACTSHEET,
        "evidence": EVIDENCE,
        "answer_status": AnswerStatus.ANSWERED,
        "original_output": correction(),
        "action_release": "analyst_review_required",
    }
    base.update(overrides)
    return InteractionRecord(**base)


class FeedbackEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self._interactions = api.interaction_store
        self._feedback = api.feedback_store
        api.interaction_store = InteractionStore(root / "interactions.jsonl")
        api.feedback_store = FeedbackStore(root / "feedback.jsonl")
        api.interaction_store.append(interaction())
        self.client = TestClient(api.app)

    def tearDown(self) -> None:
        api.interaction_store = self._interactions
        api.feedback_store = self._feedback
        self._tmp.cleanup()

    def _post(self, **overrides):
        body = {
            "interaction_id": "IX-1",
            "error_labels": ["unsupported_claim"],
            "root_cause": "model_behaviour",
            "corrected_output": correction(),
            "quality_score": 5,
            "review_status": "approved",
        }
        body.update(overrides)
        return self.client.post("/v1/analyse/feedback", json=body)

    def test_a_valid_correction_is_recorded_as_trainable(self) -> None:
        response = self._post()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["eligible_for_training"])
        self.assertTrue(response.json()["revalidation"]["passed"])

    def test_an_unknown_interaction_is_refused(self) -> None:
        response = self._post(interaction_id="IX-does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_a_correction_citing_unseen_evidence_is_refused(self) -> None:
        # Training on this teaches the model to cite from memory, which is the failure the
        # citation guardrail exists to catch.
        response = self._post(corrected_output=correction(["SAMA-CIRC-9#1.1"]))
        self.assertEqual(response.status_code, 422)
        self.assertIn("unknown_citation:SAMA-CIRC-9#1.1", response.json()["detail"]["failures"])

    def test_a_correction_with_an_uncited_fact_is_refused(self) -> None:
        response = self._post(corrected_output=correction([]))
        self.assertEqual(response.status_code, 422)
        self.assertIn("material_fact_without_citation", response.json()["detail"]["failures"])

    def test_malformed_json_is_refused(self) -> None:
        response = self._post(corrected_output="not json at all")
        self.assertEqual(response.status_code, 422)

    def test_a_correction_that_is_not_a_credit_response_is_refused(self) -> None:
        response = self._post(corrected_output=json.dumps({"summary": "fine"}))
        self.assertEqual(response.status_code, 422)

    def test_a_model_behaviour_defect_needs_a_correction(self) -> None:
        response = self._post(corrected_output=None)
        self.assertEqual(response.status_code, 422)
        self.assertIn("corrected_output", response.json()["detail"])

    def test_a_refused_correction_is_not_written(self) -> None:
        self._post(corrected_output=correction(["SAMA-CIRC-9#1.1"]))
        self.assertEqual(api.feedback_store.read_all(), [])

    def test_a_routed_defect_is_recorded_but_not_trainable(self) -> None:
        # Retraining cannot fix a retrieval bug, so it goes to a backlog instead.
        response = self._post(
            root_cause="retrieval", error_labels=["wrong_retrieval"], corrected_output=None
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["eligible_for_training"])
        self.assertEqual(response.json()["remediation_route"], "rag_index_backlog")

    def test_the_caller_cannot_declare_a_record_trainable(self):
        self.assertEqual(self._post(eligible_for_training=True).status_code, 422)

    def test_the_record_captures_what_the_model_was_shown(self) -> None:
        self._post()
        record = api.feedback_store.read_all()[0]
        self.assertEqual(record.input_factsheet, FACTSHEET)
        self.assertEqual(record.input_evidence, EVIDENCE)
        self.assertEqual(record.input_question, "Has the obligor deteriorated?")

    def test_provenance_comes_from_the_interaction_not_the_caller(self) -> None:
        # The model and adapter that produced the answer, not whatever is configured now.
        self._post()
        record = api.feedback_store.read_all()[0]
        self.assertEqual(record.model_id, "qwen3.5-9b")
        self.assertEqual(record.adapter_version, "base")

    def test_an_abstention_can_be_corrected(self) -> None:
        # An unnecessary abstention is a model-behaviour defect like any other. If it
        # cannot be corrected, the loop only ever learns from answers the model gave.
        api.interaction_store.append(
            interaction(
                "IX-2", answer_status=AnswerStatus.INSUFFICIENT_EVIDENCE, original_output=None
            )
        )
        response = self._post(interaction_id="IX-2", error_labels=["should_have_abstained"])
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["eligible_for_training"])

    def test_the_recorded_correction_becomes_a_training_example(self) -> None:
        # The end the whole loop exists for.
        self._post()
        examples, report = build_training_batch(api.feedback_store.read_all())
        self.assertEqual(len(examples), 1)
        self.assertEqual(report["skipped_no_captured_input"], 0)
        self.assertEqual(examples[0]["messages"][0]["content"], SYSTEM_PROMPT)
        payload = json.loads(examples[0]["messages"][1]["content"])
        self.assertEqual(payload["context"]["factsheet"], FACTSHEET)


class InteractionStoreTests(unittest.TestCase):
    def test_a_missing_file_reads_as_not_found(self) -> None:
        with TemporaryDirectory() as tmp:
            store = InteractionStore(Path(tmp) / "nothing.jsonl")
            with self.assertRaises(InteractionNotFound):
                store.get("IX-1")

    def test_the_latest_record_for_an_id_wins(self) -> None:
        # A replayed interaction is corrected against what was most recently served.
        with TemporaryDirectory() as tmp:
            store = InteractionStore(Path(tmp) / "i.jsonl")
            store.append(interaction("IX-1", question="first"))
            store.append(interaction("IX-1", question="second"))
            self.assertEqual(store.get("IX-1").question, "second")

    def test_it_stores_the_factsheet_not_the_rows(self) -> None:
        # The monthly rows never leave the data service; the factsheet is what the model
        # actually saw, and it is the only thing a training example needs.
        self.assertNotIn("rows", InteractionRecord.model_fields)
        self.assertIn("factsheet", InteractionRecord.model_fields)


class DuplicateCorrectionTests(unittest.TestCase):
    def test_two_corrections_on_one_interaction_train_once(self) -> None:
        # Both are kept in the log for audit, but they shared an example_id, so the batch
        # silently weighted that interaction twice.
        from credit_risk.schemas import FeedbackRecord

        def record(corrected: str) -> FeedbackRecord:
            return FeedbackRecord(
                interaction_id="IX-1",
                model_id="qwen",
                adapter_version="base",
                dataset_version="none",
                portfolio=Portfolio.CORPORATE,
                task_type="credit_deterioration",
                input_case_id=FACTSHEET["case_id"],
                input_question="q",
                input_factsheet=FACTSHEET,
                input_evidence=EVIDENCE,
                original_output="wrong",
                error_labels=["unsupported_claim"],
                corrected_output=corrected,
                root_cause="model_behaviour",
                reviewer_id="synthetic-reviewer",
                review_status="approved",
                quality_score=5,
                eligible_for_training=True,
            )

        examples, report = build_training_batch(
            [
                record(correction()),
                record(correction(executive_summary="current_position.days_past_due = 0.")),
            ]
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(report["superseded_by_later_correction"], 1)

    def test_the_later_correction_wins(self) -> None:
        from credit_risk.schemas import FeedbackRecord

        def record(corrected: str) -> FeedbackRecord:
            return FeedbackRecord(
                interaction_id="IX-1",
                model_id="qwen",
                adapter_version="base",
                dataset_version="none",
                portfolio=Portfolio.CORPORATE,
                task_type="credit_deterioration",
                input_case_id=FACTSHEET["case_id"],
                input_question="q",
                input_factsheet=FACTSHEET,
                input_evidence=EVIDENCE,
                original_output="wrong",
                error_labels=["unsupported_claim"],
                corrected_output=corrected,
                root_cause="model_behaviour",
                reviewer_id="synthetic-reviewer",
                review_status="approved",
                quality_score=5,
                eligible_for_training=True,
            )

        examples, _ = build_training_batch(
            [
                record(correction()),
                record(correction(executive_summary="current_position.days_past_due = 0.")),
            ]
        )
        self.assertEqual(
            examples[0]["messages"][-1]["content"],
            correction(executive_summary="current_position.days_past_due = 0."),
        )


class CapabilityTests(unittest.TestCase):
    def test_the_api_may_record_but_not_read_feedback(self) -> None:
        # The service that produced an answer must not also decide what is trainable.
        from credit_risk.architecture_policy import ArchitecturePolicy, ArchitecturePolicyError

        policy = ArchitecturePolicy(
            Path(__file__).parents[1] / "configs" / "architecture_policy.yaml", "1.2.0"
        )
        policy.require("api_gateway", "record_analysis_feedback")
        policy.require("api_gateway", "record_interaction")
        with self.assertRaises(ArchitecturePolicyError):
            policy.require("api_gateway", "read_feedback")
        with self.assertRaises(ArchitecturePolicyError):
            policy.require("api_gateway", "write_training_batch")

    def test_the_worker_may_read_but_not_serve(self) -> None:
        from credit_risk.architecture_policy import ArchitecturePolicy, ArchitecturePolicyError

        policy = ArchitecturePolicy(
            Path(__file__).parents[1] / "configs" / "architecture_policy.yaml", "1.2.0"
        )
        policy.require("feedback_worker", "read_feedback")
        with self.assertRaises(ArchitecturePolicyError):
            policy.require("feedback_worker", "call_model")


if __name__ == "__main__":
    unittest.main()
