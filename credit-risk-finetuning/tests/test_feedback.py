import json
import unittest

from credit_risk.feedback import build_training_batch
from credit_risk.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from credit_risk.schemas import FeedbackRecord, Portfolio

FACTSHEET = {
    "case_id": "CASE-OBL-0008-2026-01-15",
    "obligor_id": "OBL-0008",
    "portfolio": "sme",
    "jurisdiction": "SAMA",
    "current_position": {"stage": 1, "days_past_due": 0},
}
EVIDENCE = [{"evidence_id": "SAMA-CIRC-4#7.2", "jurisdiction": "SAMA",
             "text": "Stage 2 on a significant increase in credit risk."}]


def record(interaction_id: str, root_cause: str = "model_behaviour",
           with_input: bool = True, **overrides) -> FeedbackRecord:
    base = dict(
        interaction_id=interaction_id, model_id="qwen", adapter_version="v1",
        dataset_version="d1", portfolio=Portfolio.SME, task_type="ews",
        input_case_id=f"case{interaction_id}", original_output="wrong",
        error_labels=["unsupported_claim"], corrected_output="corrected",
        root_cause=root_cause, eligible_for_training=True,
    )
    if with_input:
        base.update(input_question="Has the obligor deteriorated?",
                    input_factsheet=FACTSHEET, input_evidence=EVIDENCE)
    base.update(overrides)
    return FeedbackRecord(**base)


class FeedbackTests(unittest.TestCase):
    def test_only_model_behaviour_with_correction_enters_training(self) -> None:
        examples, report = build_training_batch(
            [record("1"), record("2", root_cause="retrieval",
                                 error_labels=["wrong_retrieval"])])
        self.assertEqual(len(examples), 1)
        self.assertEqual(report["root_causes"]["retrieval"], 1)


class ReconstructedInputTests(unittest.TestCase):
    """A feedback example has to carry the input the model was actually shown.

    The batch builder used to emit "Case reference: FB-123" as the user turn, with no
    system prompt and none of the case data. Training on that teaches the model to produce
    a full credit assessment from an identifier - which is training it to invent.
    """

    def test_the_example_carries_the_shared_system_prompt(self) -> None:
        examples, _ = build_training_batch([record("1")])
        self.assertEqual(examples[0]["messages"][0]["role"], "system")
        self.assertEqual(examples[0]["messages"][0]["content"], SYSTEM_PROMPT)

    def test_the_user_turn_carries_the_factsheet_and_evidence(self) -> None:
        examples, _ = build_training_batch([record("1")])
        payload = json.loads(examples[0]["messages"][1]["content"])
        self.assertEqual(payload["context"]["factsheet"], FACTSHEET)
        self.assertEqual(payload["context"]["evidence"], EVIDENCE)
        self.assertEqual(payload["question"], "Has the obligor deteriorated?")

    def test_the_user_turn_is_not_a_bare_case_reference(self) -> None:
        examples, _ = build_training_batch([record("1")])
        self.assertNotIn("Case reference:", examples[0]["messages"][1]["content"])

    def test_a_record_without_captured_input_is_skipped_not_degraded(self) -> None:
        examples, report = build_training_batch([record("1", with_input=False)])
        self.assertEqual(examples, [])
        self.assertEqual(report["skipped_no_captured_input"], 1)

    def test_skipped_records_are_reported_not_silent(self) -> None:
        # A correction a reviewer took the trouble to write, that cannot be trained on, is
        # a defect in what the API captured - it must not vanish.
        _, report = build_training_batch(
            [record("1"), record("2", with_input=False)])
        self.assertEqual(report["training_examples"], 1)
        self.assertEqual(report["skipped_no_captured_input"], 1)

    def test_the_prompt_version_is_recorded(self) -> None:
        # An adapter trained under one prompt version is not comparable with one trained
        # under another, so the batch has to say which it used.
        examples, _ = build_training_batch([record("1")])
        self.assertEqual(examples[0]["prompt_version"], PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
