from datetime import date
from pathlib import Path
import unittest

from credit_risk.query_guard import GuardedQueryCompiler, QueryGuardError, SchemaRegistry
from credit_risk.schemas import Jurisdiction, Portfolio, QueryPlan


REGISTRY = Path(__file__).parents[1] / "configs" / "schema_registry.yaml"


class QueryGuardTests(unittest.TestCase):
    def test_parameterised_query_contains_no_identifier_values(self) -> None:
        plan = QueryPlan(
            portfolio=Portfolio.CORPORATE,
            jurisdiction=Jurisdiction.SAMA,
            obligor_id="OBL-123",
            date_from=date(2025, 1, 31),
            date_to=date(2025, 12, 31),
            metrics=["current_ratio", "pit_pd"],
        )
        compiled = GuardedQueryCompiler(SchemaRegistry(REGISTRY)).compile(plan)
        self.assertNotIn("OBL-123", compiled.sql)
        self.assertEqual(compiled.parameters[0], "OBL-123")
        self.assertIn("current_assets", compiled.selected_columns)

    def test_retail_cannot_request_corporate_ratio(self) -> None:
        plan = QueryPlan(
            portfolio=Portfolio.RETAIL,
            jurisdiction=Jurisdiction.CBUAE,
            obligor_id="OBL-1",
            date_from=date(2025, 1, 31),
            date_to=date(2025, 2, 28),
            metrics=["current_ratio"],
        )
        with self.assertRaises(QueryGuardError):
            GuardedQueryCompiler(SchemaRegistry(REGISTRY)).compile(plan)


if __name__ == "__main__":
    unittest.main()

