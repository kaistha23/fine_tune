"""Portfolio-level cohort analysis (finding M4).

The compiler only ever emitted `obligor_id = ?`, so portfolio questions - stage migration,
average PD by rating - were unreachable. Aggregation opens a disclosure path the
obligor-scoped design never had, so most of these tests are about what must be refused.
"""
import unittest
from datetime import date
from pathlib import Path

from credit_risk.data_service import ResultValidationError, validate_result
from credit_risk.query_guard import GuardedQueryCompiler, QueryGuardError, SchemaRegistry
from credit_risk.schemas import EntityLevel, Jurisdiction, Portfolio, QueryPlan

REGISTRY = Path(__file__).parents[1] / "configs" / "schema_registry.yaml"


def compiler() -> GuardedQueryCompiler:
    return GuardedQueryCompiler(SchemaRegistry(REGISTRY))


def controls() -> dict:
    return SchemaRegistry(REGISTRY).data["query_controls"]


def cohort_plan(**overrides) -> QueryPlan:
    base = dict(
        portfolio=Portfolio.SME, jurisdiction=Jurisdiction.SAMA,
        entity_level=EntityLevel.PORTFOLIO, group_by=["observation_date"],
        date_from=date(2025, 1, 31), date_to=date(2025, 12, 31),
        as_of_date=date(2026, 1, 15), metrics=["pit_pd"],
    )
    base.update(overrides)
    return QueryPlan(**base)


class CohortPlanValidationTests(unittest.TestCase):
    def test_a_cohort_plan_must_not_name_an_obligor(self) -> None:
        # Otherwise a per-obligor extract is reachable by adding a GROUP BY, and the
        # cohort-size floor never binds.
        with self.assertRaises(ValueError):
            cohort_plan(obligor_id="OBL-0008")

    def test_a_cohort_plan_must_not_name_a_facility(self) -> None:
        with self.assertRaises(ValueError):
            cohort_plan(facility_id="FAC-1")

    def test_an_obligor_plan_still_requires_an_obligor_id(self) -> None:
        with self.assertRaises(ValueError):
            QueryPlan(portfolio=Portfolio.SME, jurisdiction=Jurisdiction.SAMA,
                      date_from=date(2025, 1, 31), date_to=date(2025, 12, 31),
                      as_of_date=date(2026, 1, 15), metrics=["pit_pd"])

    def test_group_by_is_rejected_on_an_obligor_plan(self) -> None:
        with self.assertRaises(ValueError):
            QueryPlan(portfolio=Portfolio.SME, jurisdiction=Jurisdiction.SAMA,
                      obligor_id="OBL-0008", group_by=["stage"],
                      date_from=date(2025, 1, 31), date_to=date(2025, 12, 31),
                      as_of_date=date(2026, 1, 15), metrics=["pit_pd"])


class CohortCompilationTests(unittest.TestCase):
    def test_it_groups_and_aggregates(self) -> None:
        compiled = compiler().compile(cohort_plan())
        self.assertIn("GROUP BY", compiled.sql)
        self.assertIn('AVG("pit_pd") AS "pit_pd"', compiled.sql)
        self.assertEqual(compiled.aggregations, {"pit_pd": "avg(pit_pd)"})

    def test_it_never_selects_or_filters_an_obligor(self) -> None:
        compiled = compiler().compile(cohort_plan())
        self.assertNotIn("obligor_id", compiled.selected_columns)
        self.assertNotIn("obligor_id = <masked>", compiled.filters)
        self.assertNotIn("obligor_id", compiled.group_by)
        # The identifier may appear only inside COUNT(DISTINCT ...), never as a projected
        # value, a predicate or a grouping key.
        without_count = compiled.sql.replace('COUNT(DISTINCT "obligor_id")', "")
        self.assertNotIn("obligor_id", without_count)

    def test_portfolio_and_jurisdiction_are_always_grouped(self) -> None:
        # Every returned row must carry them, or the data service has nothing to assert
        # the jurisdiction separation against.
        compiled = compiler().compile(cohort_plan())
        self.assertEqual(compiled.group_by[:2], ["portfolio", "jurisdiction"])

    def test_the_cohort_size_floor_is_applied_in_sql(self) -> None:
        compiled = compiler().compile(cohort_plan())
        self.assertIn('HAVING COUNT(DISTINCT "obligor_id") >= ?', compiled.sql)
        self.assertEqual(compiled.minimum_cohort_size, 25)
        self.assertEqual(compiled.parameters[-1], 25)

    def test_cohort_size_is_always_projected(self) -> None:
        # It is what lets the data service re-check the floor on the rows returned.
        compiled = compiler().compile(cohort_plan())
        self.assertIn('COUNT(DISTINCT "obligor_id") AS cohort_size', compiled.sql)
        self.assertIn("cohort_size", compiled.selected_columns)

    def test_the_floor_counts_borrowers_not_rows(self) -> None:
        # The table is obligor-month grain. COUNT(*) over a group spanning a year reaches
        # 25 with three borrowers in it, so a row count is not a k-anonymity floor.
        compiled = compiler().compile(cohort_plan(group_by=["stage"]))
        self.assertNotIn("COUNT(*)", compiled.sql)
        self.assertIn("COUNT(DISTINCT", compiled.sql)

    def test_point_in_time_predicates_still_apply(self) -> None:
        compiled = compiler().compile(cohort_plan())
        self.assertIn('"data_cutoff_date" <= ?', compiled.sql)
        self.assertIn('"model_run_date" <= ?', compiled.sql)
        self.assertIn("data_cutoff_date", compiled.point_in_time_columns)

    def test_the_result_grain_is_the_group_by_tuple(self) -> None:
        compiled = compiler().compile(cohort_plan())
        self.assertEqual(compiled.grain, "portfolio_jurisdiction_observation_date")

    def test_an_obligor_query_is_unchanged(self) -> None:
        compiled = compiler().compile(QueryPlan(
            portfolio=Portfolio.SME, jurisdiction=Jurisdiction.SAMA,
            obligor_id="OBL-0008", date_from=date(2025, 1, 31),
            date_to=date(2025, 12, 31), as_of_date=date(2026, 1, 15),
            metrics=["pit_pd"]))
        self.assertIn('"obligor_id" = ?', compiled.sql)
        self.assertNotIn("GROUP BY", compiled.sql)
        self.assertIsNone(compiled.minimum_cohort_size)


class CohortDefaultDenyTests(unittest.TestCase):
    def test_an_unlisted_dimension_is_refused(self) -> None:
        with self.assertRaises(QueryGuardError):
            compiler().compile(cohort_plan(group_by=["revenue"]))

    def test_grouping_on_an_identifier_is_refused(self) -> None:
        with self.assertRaises(QueryGuardError):
            compiler().compile(cohort_plan(group_by=["obligor_id"]))

    def test_too_many_dimensions_are_refused(self) -> None:
        with self.assertRaises(QueryGuardError):
            compiler().compile(
                cohort_plan(group_by=["observation_date", "stage", "internal_rating"]))

    def test_a_governed_ratio_cannot_be_aggregated(self) -> None:
        # AVG(numerator)/AVG(denominator) is not AVG(ratio). For DSCR the two differ by
        # enough to change a credit decision, so the compiler refuses rather than
        # returning a number that looks right.
        with self.assertRaises(QueryGuardError) as caught:
            compiler().compile(cohort_plan(metrics=["dscr"]))
        self.assertIn("ratio of averages", str(caught.exception))

    def test_averaging_a_categorical_column_is_refused(self) -> None:
        with self.assertRaises(QueryGuardError) as caught:
            compiler().compile(cohort_plan(metrics=["internal_rating"]))
        self.assertIn("string", str(caught.exception))

    def test_min_and_max_are_not_available(self) -> None:
        # Both return one obligor's actual value however large the cohort is, so they are
        # absent from the registry allowlist rather than merely discouraged.
        allowed = controls()["cohort"]["allowed_aggregations"]
        self.assertNotIn("min", allowed)
        self.assertNotIn("max", allowed)

    def test_counting_a_categorical_column_is_allowed(self) -> None:
        compiled = compiler().compile(
            cohort_plan(metrics=["internal_rating"], cohort_aggregation="count"))
        self.assertIn('COUNT("internal_rating")', compiled.sql)


class CohortResultValidationTests(unittest.TestCase):
    def _rows(self, **overrides):
        row = dict(portfolio="sme", jurisdiction="SAMA", observation_date="2025-06-30",
                   cohort_size=40, pit_pd=0.04, data_cutoff_date="2025-12-31",
                   model_run_date="2025-12-31")
        row.update(overrides)
        return [row]

    def test_a_valid_cohort_result_passes(self) -> None:
        plan = cohort_plan()
        compiled = compiler().compile(plan)
        report = validate_result(self._rows(), compiled, plan, controls())
        self.assertTrue(report["passed"])
        self.assertEqual(report["grain_keys"], compiled.group_by)

    def test_a_cell_below_the_floor_is_refused(self) -> None:
        # Defence in depth: this must not depend on the HAVING clause having compiled
        # correctly, because the consequence is an individual disclosure.
        plan = cohort_plan()
        compiled = compiler().compile(plan)
        with self.assertRaises(ResultValidationError) as caught:
            validate_result(self._rows(cohort_size=3), compiled, plan, controls())
        self.assertIn("cohort_below_minimum", str(caught.exception))

    def test_a_missing_cohort_size_is_refused(self) -> None:
        plan = cohort_plan()
        compiled = compiler().compile(plan)
        rows = self._rows()
        del rows[0]["cohort_size"]
        with self.assertRaises(ResultValidationError):
            validate_result(rows, compiled, plan, controls())

    def test_a_leaked_identifier_is_refused(self) -> None:
        plan = cohort_plan()
        compiled = compiler().compile(plan)
        with self.assertRaises(ResultValidationError) as caught:
            validate_result(self._rows(obligor_id="OBL-0008"), compiled, plan, controls())
        self.assertIn("cohort_leaked_identifier", str(caught.exception))

    def test_a_point_in_time_breach_is_still_caught(self) -> None:
        plan = cohort_plan()
        compiled = compiler().compile(plan)
        with self.assertRaises(ResultValidationError) as caught:
            validate_result(self._rows(model_run_date="2026-06-30"), compiled, plan,
                            controls())
        self.assertIn("point_in_time_breach", str(caught.exception))

    def test_a_jurisdiction_mismatch_is_still_caught(self) -> None:
        plan = cohort_plan()
        compiled = compiler().compile(plan)
        with self.assertRaises(ResultValidationError):
            validate_result(self._rows(jurisdiction="CBUAE"), compiled, plan, controls())


class CohortFactsheetTests(unittest.TestCase):
    def test_a_cohort_cannot_become_a_factsheet(self) -> None:
        # The factsheet is obligor-level throughout, and every consumer of it reasons
        # about one borrower. Without this guard a cohort plan reached CreditFactsheet
        # with obligor_id=None and surfaced a raw pydantic error to the caller.
        from credit_risk.factsheet import FactsheetError, build_factsheet
        rows = [dict(portfolio="sme", jurisdiction="SAMA",
                     observation_date="2025-06-30", cohort_size=30, pit_pd=0.04)]
        with self.assertRaises(FactsheetError) as caught:
            build_factsheet(rows, cohort_plan())
        self.assertIn("/v1/query/fetch", str(caught.exception))

    def test_an_obligor_factsheet_still_builds(self) -> None:
        from credit_risk.factsheet import build_factsheet
        plan = QueryPlan(
            portfolio=Portfolio.SME, jurisdiction=Jurisdiction.SAMA,
            obligor_id="OBL-0008", date_from=date(2025, 1, 31),
            date_to=date(2025, 12, 31), as_of_date=date(2026, 1, 15), metrics=["pit_pd"])
        rows = [dict(obligor_id="OBL-0008", portfolio="sme", jurisdiction="SAMA",
                     observation_date="2025-06-30", pit_pd=0.04)]
        self.assertEqual(build_factsheet(rows, plan).obligor_id, "OBL-0008")


class CohortEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from credit_risk.api import app
        self.client = TestClient(app)
        self.body = {
            "user_text": "Average PD for the SME book through 2025",
            "query_plan": {
                "portfolio": "sme", "jurisdiction": "SAMA", "entity_level": "portfolio",
                "group_by": ["observation_date"], "date_from": "2025-01-01",
                "date_to": "2025-12-31", "as_of_date": "2026-01-15",
                "metrics": ["pit_pd"],
            },
        }

    def test_validate_accepts_a_cohort_plan(self) -> None:
        response = self.client.post("/v1/query/validate", json=self.body)
        self.assertEqual(response.status_code, 200)
        review = response.json()["sql_review"]
        self.assertEqual(review["group_by"],
                         ["portfolio", "jurisdiction", "observation_date"])
        self.assertEqual(review["minimum_cohort_size"], 25)
        self.assertEqual(review["aggregations"], {"pit_pd": "avg(pit_pd)"})

    def test_the_reviewer_sees_the_suppression_floor(self) -> None:
        # "grain" alone tells a reviewer nothing about whether an aggregate is releasable.
        review = self.client.post(
            "/v1/query/validate", json=self.body).json()["sql_review"]
        self.assertIn("cohort_size >= 25 (small cells suppressed)", review["filters"])

    def test_factsheet_and_analyse_refuse_before_fetching_rows(self) -> None:
        # Refusing after the fetch would pull obligor data across the service boundary for
        # a request that cannot succeed. No data service is running in this test, so a 422
        # rather than a 502 is what proves the check happens first.
        for route in ("/v1/factsheet", "/v1/analyse"):
            with self.subTest(route=route):
                response = self.client.post(route, json=self.body)
                self.assertEqual(response.status_code, 422)
                self.assertIn("/v1/query/fetch", response.json()["detail"])


class TruncationTests(unittest.TestCase):
    """The row-limit check could never fire: SQL asked for exactly the limit."""

    def test_the_compiler_asks_for_one_row_over_the_limit(self) -> None:
        limit = int(controls()["maximum_result_rows"])
        compiled = compiler().compile(cohort_plan())
        self.assertIn(f"LIMIT {limit + 1}", compiled.sql)

    def test_an_over_long_result_is_reported_as_truncated(self) -> None:
        plan = cohort_plan()
        compiled = compiler().compile(plan)
        limit = int(controls()["maximum_result_rows"])
        rows = [
            dict(portfolio="sme", jurisdiction="SAMA",
                 observation_date=f"2025-06-{(i % 28) + 1:02d}", cohort_size=40,
                 pit_pd=0.04, data_cutoff_date="2025-01-01", model_run_date="2025-01-01")
            for i in range(limit + 1)
        ]
        with self.assertRaises(ResultValidationError) as caught:
            validate_result(rows, compiled, plan, controls())
        self.assertIn("result_truncated", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
