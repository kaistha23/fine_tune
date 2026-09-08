"""The human SQL review loop end to end (finding H5).

The handover repository produced a review packet and then had nowhere to send a decision:
no endpoint, no persistence, no revalidation of corrections, and build_training_batch
dropped every SQL-review record on the floor.
"""
import tempfile
import unittest
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from credit_risk import api
from credit_risk.api import app, build_sql_review_packet, compiler
from credit_risk.feedback import build_training_batch
from credit_risk.feedback_store import FeedbackStore
from credit_risk.schemas import Jurisdiction, Portfolio, QueryPlan


def plan(**overrides) -> QueryPlan:
    base = dict(
        portfolio=Portfolio.CORPORATE, jurisdiction=Jurisdiction.SAMA,
        obligor_id="OBL-0008", date_from=date(2025, 1, 31), date_to=date(2025, 12, 31),
        as_of_date=date(2026, 1, 15), metrics=["current_ratio", "pit_pd"],
    )
    base.update(overrides)
    return QueryPlan(**base)


class ReviewPacketTests(unittest.TestCase):
    def test_packet_shows_everything_the_reviewer_must_check(self) -> None:
        packet = build_sql_review_packet(compiler.compile(plan()))
        # The mandate: exact SQL, hash, schema version, grain, tables, columns, joins,
        # filters and governed calculations.
        for key in ("parameterised_sql", "query_hash", "schema_registry_version", "grain",
                    "source_table", "selected_columns", "joins", "filters",
                    "governed_calculations", "point_in_time_columns"):
            self.assertIn(key, packet)
        self.assertEqual(packet["grain"], "obligor_month")
        self.assertEqual(packet["governed_calculations"]["current_ratio"],
                         "ratio.current_ratio.v1")
        self.assertIn("data_cutoff_date", packet["point_in_time_columns"])

    def test_values_stay_masked_and_sql_is_not_executable_by_the_reviewer(self) -> None:
        packet = build_sql_review_packet(compiler.compile(plan(obligor_id="SECRET-OBLIGOR")))
        self.assertNotIn("SECRET-OBLIGOR", packet["parameterised_sql"])
        self.assertNotIn("SECRET-OBLIGOR", str(packet["parameter_values"]))
        self.assertNotIn("SECRET-OBLIGOR", str(packet["filters"]))
        self.assertFalse(packet["editable"])
        self.assertEqual(packet["execution_status"], "NOT_EXECUTED")


class ReviewEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store_path = Path(self.tmp.name) / "feedback.jsonl"
        self._original = api.feedback_store
        api.feedback_store = FeedbackStore(self.store_path)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        api.feedback_store = self._original
        self.tmp.cleanup()

    def review_body(self, **overrides) -> dict:
        reviewed = plan()
        packet = build_sql_review_packet(compiler.compile(reviewed))
        body = {
            "interaction_id": "INT-0001",
            "query_hash": packet["query_hash"],
            "reviewed_query_plan": reviewed.model_dump(mode="json"),
            "decision": "approved",
        }
        body.update(overrides)
        return body

    def test_approval_is_persisted(self) -> None:
        response = self.client.post("/v1/query/review", json=self.review_body())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["sql_review_status"], "approved")

        stored = FeedbackStore(self.store_path).read_all()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].sql_review_status, "approved")
        self.assertEqual(len(stored[0].query_hash), 64)

    def test_rejection_requires_a_comment(self) -> None:
        response = self.client.post("/v1/query/review",
                                    json=self.review_body(decision="rejected"))
        self.assertEqual(response.status_code, 422)

    def test_rejection_with_correction_is_revalidated_and_stored(self) -> None:
        corrected = plan(metrics=["current_ratio", "dscr", "pit_pd"])
        response = self.client.post("/v1/query/review", json=self.review_body(
            decision="rejected",
            comment="dscr was missing from the deterioration view",
            error_labels=["wrong_column"],
            root_cause="schema",
            corrected_query_plan=corrected.model_dump(mode="json"),
        ))
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["revalidated"])
        self.assertIn("debt_service", body["corrected_sql_review"]["selected_columns"])
        # The corrected template is a fresh compilation, so it gets its own hash.
        self.assertNotEqual(body["corrected_sql_review"]["query_hash"],
                            self.review_body()["query_hash"])

        stored = FeedbackStore(self.store_path).read_all()
        self.assertEqual(stored[0].sql_review_status, "rejected")
        self.assertIsNotNone(stored[0].corrected_query_plan)
        self.assertFalse(stored[0].eligible_for_training)

    def test_a_correction_that_breaks_the_rules_is_refused(self) -> None:
        # Reviewer "corrects" to a metric that is not approved for this grain. The
        # correction goes through the same default-deny cycle and is refused.
        bad = plan(entity_level="facility", facility_id="FAC-0008-1",
                   metrics=["current_ratio"])
        response = self.client.post("/v1/query/review", json=self.review_body(
            decision="rejected", comment="try facility grain",
            corrected_query_plan=bad.model_dump(mode="json"),
        ))
        self.assertEqual(response.status_code, 422)
        self.assertIn("failed revalidation", response.json()["detail"])

    def test_review_of_a_plan_the_reviewer_was_not_shown_is_refused(self) -> None:
        response = self.client.post("/v1/query/review",
                                    json=self.review_body(query_hash="0" * 64))
        self.assertEqual(response.status_code, 409)


class TriageRoutingTests(unittest.TestCase):
    def test_sql_review_records_are_routed_not_dropped(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            store = FeedbackStore(Path(tmp.name) / "feedback.jsonl")
            api_store, api.feedback_store = api.feedback_store, store
            client = TestClient(app)
            reviewed = plan()
            packet = build_sql_review_packet(compiler.compile(reviewed))
            client.post("/v1/query/review", json={
                "interaction_id": "INT-0002",
                "query_hash": packet["query_hash"],
                "reviewed_query_plan": reviewed.model_dump(mode="json"),
                "decision": "rejected",
                "comment": "wrong grain",
                "error_labels": ["wrong_grain"],
                "root_cause": "schema",
            })
            api.feedback_store = api_store

            examples, report = build_training_batch(store.read_all())
            # Not training data, but accounted for and owned by a queue.
            self.assertEqual(examples, [])
            self.assertEqual(report["remediation_routes"]["schema_and_query_backlog"], 1)
            self.assertEqual(report["sql_reviews"]["rejected"], 1)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
