"""The human SQL review loop end to end (finding H5).

The handover repository produced a review packet and then had nowhere to send a decision:
no endpoint, no persistence, no revalidation of corrections, and build_training_batch
dropped every SQL-review record on the floor.
"""

import unittest
from datetime import date

from credit_risk.data_service import build_sql_review_packet, compiler
from credit_risk.schemas import Jurisdiction, Portfolio, QueryPlan


def plan(**overrides) -> QueryPlan:
    base = {
        "portfolio": Portfolio.CORPORATE,
        "jurisdiction": Jurisdiction.SAMA,
        "obligor_id": "OBL-0008",
        "date_from": date(2025, 1, 31),
        "date_to": date(2025, 12, 31),
        "as_of_date": date(2026, 1, 15),
        "metrics": ["current_ratio", "pit_pd"],
    }
    base.update(overrides)
    return QueryPlan(**base)


class ReviewPacketTests(unittest.TestCase):
    def test_packet_shows_everything_the_reviewer_must_check(self) -> None:
        packet = build_sql_review_packet(compiler.compile(plan()))
        # The mandate: exact SQL, hash, schema version, grain, tables, columns, joins,
        # filters and governed calculations.
        for key in (
            "parameterised_sql",
            "query_hash",
            "schema_registry_version",
            "grain",
            "source_table",
            "selected_columns",
            "joins",
            "filters",
            "governed_calculations",
            "point_in_time_columns",
        ):
            self.assertIn(key, packet)
        self.assertEqual(packet["grain"], "obligor_month")
        self.assertEqual(packet["governed_calculations"]["current_ratio"], "ratio.current_ratio.v2")
        self.assertIn("data_cutoff_date", packet["point_in_time_columns"])

    def test_values_stay_masked_and_sql_is_not_executable_by_the_reviewer(self) -> None:
        packet = build_sql_review_packet(compiler.compile(plan(obligor_id="SECRET-OBLIGOR")))
        self.assertNotIn("SECRET-OBLIGOR", packet["parameterised_sql"])
        self.assertNotIn("SECRET-OBLIGOR", str(packet["parameter_values"]))
        self.assertNotIn("SECRET-OBLIGOR", str(packet["filters"]))
        self.assertFalse(packet["editable"])
        self.assertEqual(packet["execution_status"], "NOT_EXECUTED")


# Transactional endpoint coverage moved to test_secure_review.py.
