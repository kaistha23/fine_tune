import unittest
from datetime import date

from credit_risk.api import build_sql_review_packet, compiler
from credit_risk.schemas import Jurisdiction, Portfolio, QueryPlan


class SqlReviewTests(unittest.TestCase):
    def test_review_packet_exposes_template_but_masks_values(self) -> None:
        plan = QueryPlan(
            portfolio=Portfolio.CORPORATE,
            jurisdiction=Jurisdiction.SAMA,
            obligor_id="CONFIDENTIAL-OBLIGOR",
            date_from=date(2025, 1, 31),
            date_to=date(2025, 12, 31),
            as_of_date=date(2025, 12, 31),
            metrics=["current_ratio", "pit_pd"],
        )
        packet = build_sql_review_packet(compiler.compile(plan))
        self.assertIn("SELECT", packet["parameterised_sql"])
        self.assertNotIn("CONFIDENTIAL-OBLIGOR", packet["parameterised_sql"])
        self.assertNotIn("CONFIDENTIAL-OBLIGOR", str(packet["parameter_values"]))
        self.assertFalse(packet["editable"])
        self.assertEqual(packet["execution_status"], "NOT_EXECUTED")
        self.assertEqual(len(packet["query_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
