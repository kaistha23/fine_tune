import json
import os
import pathlib
import tempfile
import unittest

from credit_risk.dataset import SPLITS, stable_split
from credit_risk.feedback import build_training_batch
from credit_risk.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from credit_risk.schemas import FeedbackRecord, Portfolio

FACTSHEET = {
    "case_id": "CASE-OBL-0008-2026-01-15",
    "obligor_id": "OBL-0008",
    "group_id": "OBL-0008",
    "as_of_date": "2026-01-15",
    "portfolio": "sme",
    "jurisdiction": "SAMA",
    "current_position": {"stage": 1, "days_past_due": 0},
}
EVIDENCE = [
    {
        "evidence_id": "SAMA-CIRC-4#7.2",
        "jurisdiction": "SAMA",
        "text": "Stage 2 on a significant increase in credit risk.",
        "document_id": "SAMA-CIRC-4",
        "document_version": "2.0",
        "section": "7.2",
        "score": 0.9,
    }
]


def record(
    interaction_id: str, root_cause: str = "model_behaviour", with_input: bool = True, **overrides
) -> FeedbackRecord:
    base = {
        "interaction_id": interaction_id,
        "model_id": "qwen",
        "adapter_version": "v1",
        "dataset_version": "d1",
        "portfolio": Portfolio.SME,
        "task_type": "ews",
        "input_case_id": f"case{interaction_id}",
        "original_output": "wrong",
        "error_labels": ["unsupported_claim"],
        "corrected_output": json.dumps(
            {"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}
        ),
        "root_cause": root_cause,
        "reviewer_id": "synthetic-reviewer",
        "review_status": "approved",
        "quality_score": 5,
        "eligible_for_training": True,
    }
    if with_input:
        base.update(
            input_question="Has the obligor deteriorated?",
            input_factsheet=FACTSHEET,
            input_evidence=EVIDENCE,
        )
    base.update(overrides)
    return FeedbackRecord(**base)


class FeedbackTests(unittest.TestCase):
    def test_only_model_behaviour_with_correction_enters_training(self) -> None:
        examples, report = build_training_batch(
            [record("1"), record("2", root_cause="retrieval", error_labels=["wrong_retrieval"])]
        )
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
        _, report = build_training_batch([record("1"), record("2", with_input=False)])
        self.assertEqual(report["training_examples"], 1)
        self.assertEqual(report["skipped_no_captured_input"], 1)

    def test_the_prompt_version_is_recorded(self) -> None:
        # An adapter trained under one prompt version is not comparable with one trained
        # under another, so the batch has to say which it used.
        examples, _ = build_training_batch([record("1")])
        self.assertEqual(examples[0]["prompt_version"], PROMPT_VERSION)


class BatchLayoutTests(unittest.TestCase):
    """The worker's output must be mergeable with the seed dataset, and trainable as-is."""

    def _write_batch(self, directory) -> dict:
        import subprocess
        import sys

        feedback = directory / "feedback.jsonl"
        feedback.write_text(record("1").model_dump_json() + "\n", encoding="utf-8")
        env = dict(os.environ, CR_SERVICE_ROLE="feedback_worker")
        (directory / "exclusions.json").write_text('{"groups":[],"content_hashes":[]}')
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "credit_risk.feedback",
                str(feedback),
                str(directory / "batch"),
                "--out-of-time-from",
                "2027-01-01",
                "--exclusions",
                str(directory / "exclusions.json"),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return json.loads(result.stdout)

    def test_it_writes_the_same_splits_as_the_seed_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._write_batch(directory)
            for split in SPLITS:
                self.assertTrue((directory / "batch" / f"{split}.jsonl").is_file())
                self.assertTrue((directory / "batch" / f"{split}.provenance.jsonl").is_file())

    def test_the_training_line_carries_only_messages(self) -> None:
        # mlx-lm reads each line as a training record and unknown keys are not guaranteed
        # to be ignored, which is why provenance goes to a sidecar. The feedback batch was
        # writing example_id, portfolio, task_type, source and prompt_version inline.
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._write_batch(directory)
            for split in SPLITS:
                for line in (directory / "batch" / f"{split}.jsonl").read_text().splitlines():
                    if line.strip():
                        self.assertEqual(sorted(json.loads(line)), ["messages"])

    def test_provenance_lands_in_the_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            self._write_batch(directory)
            lines = [
                line
                for split in SPLITS
                for line in (directory / "batch" / f"{split}.provenance.jsonl")
                .read_text()
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 1)
            self.assertIn("example_id", json.loads(lines[0]))

    def test_an_obligor_cannot_span_train_and_test_across_producers(self) -> None:
        # Both producers split on the same obligor hash. If they split independently, a
        # borrower corrected in production could sit in train from one file and test from
        # the other, and the test score would be measured on data already trained on.
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            report = self._write_batch(directory)
            expected = stable_split(FACTSHEET["obligor_id"])
            self.assertEqual(report["splits"][expected], 1)
            for split in SPLITS:
                if split != expected:
                    self.assertEqual(report["splits"][split], 0)


if __name__ == "__main__":
    unittest.main()
