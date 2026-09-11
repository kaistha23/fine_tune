"""Training and inference must show the model the same thing.

There were three prompt definitions and they disagreed: dataset.py had its own wording and
passed the bare case as the user turn, omlx_client.py sent {question, context,
response_schema} under different wording, and feedback.py sent no system prompt and an
identifier string.

With `mask_prompt: true` the loss sits entirely on the assistant turn, so a mismatch does
not fail loudly - the run completes, the loss falls, and the adapter has learned to emit a
target conditioned on an input structure production never sends. That is a whole training
run wasted with nothing to show for it, which is why this is asserted rather than left to
review.
"""

import json
import unittest
from unittest import mock

from credit_risk.dataset import DEFAULT_QUESTION, build_sft_record
from credit_risk.feedback import build_training_batch
from credit_risk.omlx_client import OMLXClient
from credit_risk.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_messages
from credit_risk.schemas import Evidence, FeedbackRecord, Jurisdiction, Portfolio
from credit_risk.tokenization import configure_non_thinking

FACTSHEET = {
    "case_id": "CASE-OBL-0008-2026-01-15",
    "obligor_id": "OBL-0008",
    "group_id": "OBL-0008",
    "as_of_date": "2026-01-15",
    "portfolio": "sme",
    "jurisdiction": "SAMA",
    "current_position": {"stage": 1, "days_past_due": 0},
}
EVIDENCE_DICT = [
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
QUESTION = "Has the obligor deteriorated?"


def served_request() -> dict:
    """The messages the server actually posts, captured without a live oMLX."""
    captured = {}

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer_status": "ANSWERED",
                                    "executive_summary": "s",
                                    "human_approval_required": True,
                                }
                            )
                        }
                    }
                ]
            }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None, headers=None):
            captured.update(json)
            return Response()

    with mock.patch("httpx.Client", Client):
        OMLXClient("http://x/v1", "m").generate_credit_response(
            QUESTION,
            FACTSHEET,
            [
                Evidence(
                    evidence_id="SAMA-CIRC-4#7.2",
                    jurisdiction=Jurisdiction.SAMA,
                    document_id="SAMA-CIRC-4",
                    document_version="2.0",
                    section="7.2",
                    text="Stage 2 on a significant increase in credit risk.",
                    score=0.9,
                )
            ],
        )
    return captured


def served_messages() -> list[dict]:
    return served_request()["messages"]


class SystemPromptTests(unittest.TestCase):
    def test_serving_explicitly_disables_qwen_thinking(self) -> None:
        self.assertEqual(
            served_request()["chat_template_kwargs"], {"enable_thinking": False}
        )

    def test_mlx_dataset_template_defaults_to_non_thinking(self) -> None:
        calls = []

        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                calls.append(kwargs)
                return []

        tokenizer = configure_non_thinking(Tokenizer())
        tokenizer.apply_chat_template([], tokenize=True)
        self.assertFalse(calls[0]["enable_thinking"])

    def test_training_and_inference_share_one_system_prompt(self) -> None:
        trained = build_sft_record(
            FACTSHEET,
            json.dumps({"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}),
            "ews",
        )["messages"][0]
        self.assertEqual(trained["content"], served_messages()[0]["content"])

    def test_there_is_only_one_definition_of_it(self) -> None:
        self.assertEqual(served_messages()[0]["content"], SYSTEM_PROMPT)

    def test_feedback_examples_use_it_too(self) -> None:
        examples, _ = build_training_batch(
            [
                FeedbackRecord(
                    interaction_id="1",
                    model_id="qwen",
                    adapter_version="v1",
                    dataset_version="d1",
                    portfolio=Portfolio.SME,
                    task_type="ews",
                    input_case_id="c1",
                    original_output="wrong",
                    error_labels=["unsupported_claim"],
                    corrected_output=json.dumps(
                        {"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}
                    ),
                    root_cause="model_behaviour",
                    reviewer_id="synthetic-reviewer",
                    review_status="approved",
                    quality_score=5,
                    eligible_for_training=True,
                    input_question=QUESTION,
                    input_factsheet=FACTSHEET,
                    input_evidence=EVIDENCE_DICT,
                )
            ]
        )
        self.assertEqual(examples[0]["messages"][0]["content"], SYSTEM_PROMPT)


class UserTurnShapeTests(unittest.TestCase):
    def test_both_sides_use_the_same_envelope(self) -> None:
        trained = json.loads(
            build_sft_record(
                FACTSHEET,
                json.dumps({"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}),
                "ews",
                question=QUESTION,
                evidence=EVIDENCE_DICT,
            )["messages"][1]["content"]
        )
        served = json.loads(served_messages()[1]["content"])
        self.assertEqual(set(trained), {"question", "context", "response_schema"})
        self.assertEqual(set(served), {"question", "context", "response_schema"})
        self.assertEqual(set(trained["context"]), set(served["context"]))

    def test_the_factsheet_sits_in_the_same_place(self) -> None:
        trained = json.loads(
            build_sft_record(
                FACTSHEET,
                json.dumps({"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}),
                "ews",
                question=QUESTION,
                evidence=EVIDENCE_DICT,
            )["messages"][1]["content"]
        )
        served = json.loads(served_messages()[1]["content"])
        self.assertEqual(trained["context"]["factsheet"], served["context"]["factsheet"])

    def test_the_evidence_sits_in_the_same_place(self) -> None:
        trained = json.loads(
            build_sft_record(
                FACTSHEET,
                json.dumps({"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}),
                "ews",
                question=QUESTION,
                evidence=EVIDENCE_DICT,
            )["messages"][1]["content"]
        )
        served = json.loads(served_messages()[1]["content"])
        self.assertEqual(
            [e["evidence_id"] for e in trained["context"]["evidence"]],
            [e["evidence_id"] for e in served["context"]["evidence"]],
        )

    def test_the_response_schema_matches_inference(self) -> None:
        # Training and serving must condition on the identical output contract.
        trained = json.loads(
            build_sft_record(
                FACTSHEET,
                json.dumps({"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}),
                "ews",
            )["messages"][1]["content"]
        )
        self.assertEqual(
            trained["response_schema"],
            json.loads(served_messages()[1]["content"])["response_schema"],
        )
        self.assertNotIn("title", json.dumps(trained["response_schema"]))

    def test_an_example_with_no_question_gets_the_stated_default(self) -> None:
        trained = json.loads(
            build_sft_record(
                FACTSHEET,
                json.dumps({"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}),
                "ews",
            )["messages"][1]["content"]
        )
        self.assertEqual(trained["question"], DEFAULT_QUESTION)


class ProvenanceTests(unittest.TestCase):
    def test_the_record_states_which_prompt_it_was_built_under(self) -> None:
        # An adapter trained under one prompt version is not comparable with one trained
        # under another, so a checkpoint has to be traceable to its prompt.
        self.assertEqual(
            build_sft_record(
                FACTSHEET,
                json.dumps({"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}),
                "ews",
            )["prompt_version"],
            PROMPT_VERSION,
        )

    def test_the_assistant_turn_is_last(self) -> None:
        messages = build_sft_record(
            FACTSHEET,
            json.dumps({"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}),
            "ews",
        )["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])

    def test_build_messages_is_stable(self) -> None:
        first = build_messages(QUESTION, FACTSHEET, EVIDENCE_DICT)
        second = build_messages(QUESTION, FACTSHEET, EVIDENCE_DICT)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
