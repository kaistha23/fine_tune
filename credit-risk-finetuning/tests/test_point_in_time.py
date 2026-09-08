"""Point-in-time and grain controls (findings C4 and M3).

These are the tests the handover repository did not have: nothing stopped a query from
seeing data that did not exist at the analyst's as-of date, and a grain violation was
reported as a missing-column error.
"""
from datetime import date
from pathlib import Path
import unittest

from pydantic import ValidationError

from credit_risk.query_guard import GuardedQueryCompiler, QueryGuardError, SchemaRegistry
from credit_risk.schemas import EntityLevel, Jurisdiction, Portfolio, QueryPlan


REGISTRY = Path(__file__).parents[1] / "configs" / "schema_registry.yaml"


def compiler() -> GuardedQueryCompiler:
    return GuardedQueryCompiler(SchemaRegistry(REGISTRY))


def plan(**overrides) -> QueryPlan:
    base = dict(
        portfolio=Portfolio.CORPORATE,
        jurisdiction=Jurisdiction.SAMA,
        obligor_id="OBL-0008",
        date_from=date(2025, 1, 31),
        date_to=date(2025, 12, 31),
        as_of_date=date(2025, 12, 31),
        metrics=["current_ratio"],
    )
    base.update(overrides)
    return QueryPlan(**base)


class PointInTimeTests(unittest.TestCase):
    def test_observing_past_the_as_of_date_is_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            plan(date_to=date(2025, 12, 31), as_of_date=date(2025, 6, 30))
        self.assertIn("date_to must not be after as_of_date", str(ctx.exception))

    def test_data_cutoff_predicate_is_always_emitted(self) -> None:
        compiled = compiler().compile(plan())
        self.assertIn('"data_cutoff_date" <= ?', compiled.sql)
        self.assertIn("data_cutoff_date", compiled.point_in_time_columns)
        self.assertIn(date(2025, 12, 31), compiled.parameters)

    def test_model_run_predicate_only_when_model_outputs_selected(self) -> None:
        without = compiler().compile(plan(metrics=["current_ratio"]))
        self.assertNotIn('"model_run_date" <= ?', without.sql)
        self.assertEqual(without.point_in_time_columns, ["data_cutoff_date"])

        # pit_pd is a model output, so its own run date must also be bounded.
        with_model = compiler().compile(plan(metrics=["current_ratio", "pit_pd"]))
        self.assertIn('"model_run_date" <= ?', with_model.sql)
        self.assertIn("model_run_date", with_model.point_in_time_columns)

    def test_as_of_date_is_parameterised_not_interpolated(self) -> None:
        compiled = compiler().compile(plan())
        self.assertNotIn("2025-12-31", compiled.sql)


class GrainTests(unittest.TestCase):
    def test_obligor_grain_metric_is_refused_at_facility_grain(self) -> None:
        with self.assertRaises(QueryGuardError) as ctx:
            compiler().compile(plan(
                entity_level=EntityLevel.FACILITY,
                facility_id="FAC-0008-1",
                metrics=["current_ratio"],
            ))
        message = str(ctx.exception)
        # The diagnosis must name the grain, not send the reviewer hunting for columns.
        self.assertIn("facility_month", message)
        self.assertIn("not available", message)
        self.assertNotIn("missing approved columns", message)

    def test_dual_grain_metric_compiles_at_both_grains(self) -> None:
        obligor = compiler().compile(plan(metrics=["stage", "days_past_due"]))
        self.assertEqual(obligor.grain, "obligor_month")
        facility = compiler().compile(plan(
            entity_level=EntityLevel.FACILITY,
            facility_id="FAC-0008-1",
            metrics=["stage", "days_past_due"],
        ))
        self.assertEqual(facility.grain, "facility_month")
        self.assertIn("facility_id", facility.selected_columns)


class GeneratedSqlTests(unittest.TestCase):
    def test_generated_sql_carries_no_prohibited_keyword(self) -> None:
        # current_assets contains the substring "set"; the guard must use word boundaries
        # or this legitimate query would be rejected.
        compiled = compiler().compile(plan(metrics=["current_ratio"]))
        self.assertIn("current_assets", compiled.sql)

    def test_table_must_be_allowlisted(self) -> None:
        registry = SchemaRegistry(REGISTRY)
        registry.data["query_controls"]["allowed_tables"] = ["facility_monthly"]
        with self.assertRaises(QueryGuardError) as ctx:
            GuardedQueryCompiler(registry).compile(plan())
        self.assertIn("not allowlisted", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
