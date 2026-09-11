import unittest

from credit_risk.feedback import build_training_batch
from credit_risk.schemas import FeedbackRecord, Portfolio


class FeedbackTests(unittest.TestCase):
    def test_only_model_behaviour_with_correction_enters_training(self) -> None:
        trainable = FeedbackRecord(
            interaction_id="1", model_id="qwen", adapter_version="v1", dataset_version="d1",
            portfolio=Portfolio.SME, task_type="ews", input_case_id="case1",
            original_output="wrong", error_labels=["unsupported_claim"],
            corrected_output="corrected", root_cause="model_behaviour", eligible_for_training=True,
        )
        retrieval = FeedbackRecord(
            interaction_id="2", model_id="qwen", adapter_version="v1", dataset_version="d1",
            portfolio=Portfolio.SME, task_type="policy_qa", input_case_id="case2",
            original_output="wrong", error_labels=["wrong_retrieval"],
            corrected_output="corrected", root_cause="retrieval", eligible_for_training=True,
        )
        examples, report = build_training_batch([trainable, retrieval])
        self.assertEqual(len(examples), 1)
        self.assertEqual(report["root_causes"]["retrieval"], 1)


if __name__ == "__main__":
    unittest.main()

