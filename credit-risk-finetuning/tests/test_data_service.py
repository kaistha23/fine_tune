"""Data-service execution limits and result validation (finding H6).

The handover repository executed the approved query and returned rows verbatim: no
statement timeout, no memory bound, and no check that the rows matched the grain and
filters the plan was approved under. There was also no test for this module at all.
"""

import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path

from http_helpers import TestClient, execute_reviewed

from credit_risk import data_service
from credit_risk.data_service import ResultValidationError, app, validate_result
from credit_risk.query_guard import GuardedQueryCompiler, SchemaRegistry
from credit_risk.schemas import Jurisdiction, Portfolio, QueryPlan
from credit_risk.settings import settings

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "data" / "curated" / "credit_risk.duckdb"
REGISTRY = ROOT / "configs" / "schema_registry.yaml"


def ensure_fixture() -> None:
    if not FIXTURE.is_file():
        subprocess.run(
            [sys.executable, "-m", "credit_risk.data_prep.fixture"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )


def plan(**overrides) -> QueryPlan:
    base = {
        "portfolio": Portfolio.CORPORATE,
        "jurisdiction": Jurisdiction.SAMA,
        "obligor_id": "OBL-0008",
        "date_from": date(2025, 1, 31),
        "date_to": date(2025, 12, 31),
        "as_of_date": date(2026, 1, 15),
        "metrics": ["current_ratio", "dscr", "pit_pd", "stage"],
    }
    base.update(overrides)
    return QueryPlan(**base)


class AuthTests(unittest.TestCase):
    def test_execute_requires_the_service_token(self) -> None:
        client = TestClient(app)
        response = client.post("/internal/v1/query/execute", json=plan().model_dump(mode="json"))
        self.assertEqual(response.status_code, 401)

    def test_wrong_token_is_rejected(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/internal/v1/query/execute",
            headers={"x-service-token": "not-the-token"},
            json=plan().model_dump(mode="json"),
        )
        self.assertEqual(response.status_code, 401)


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_fixture()
        # The container runs this app as CR_SERVICE_ROLE=data_service; the module default
        # is api_gateway, which is correctly denied compile_parameterised_sql.
        cls._role = data_service.settings.service_role
        data_service.settings.service_role = "data_service"
        cls.client = TestClient(app)
        cls.headers = {"x-service-token": settings.service_token}

    @classmethod
    def tearDownClass(cls) -> None:
        data_service.settings.service_role = cls._role

    def post(self, query_plan: QueryPlan):
        return execute_reviewed(self.client, query_plan.model_dump(mode="json"))

    def test_approved_query_returns_validated_rows(self) -> None:
        response = self.post(plan())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["row_count"], 12)
        self.assertEqual(body["grain"], "obligor_month")
        self.assertTrue(body["validation"]["passed"])
        self.assertEqual(body["validation"]["grain_keys"], ["obligor_id", "observation_date"])

    def test_point_in_time_bound_actually_hides_later_data(self) -> None:
        # December data lands 2026-01-05 and its model runs 2026-01-10, so an analyst
        # standing at 2025-12-31 must not see the December row.
        early = self.post(plan(as_of_date=date(2025, 12, 31), date_to=date(2025, 12, 31)))
        self.assertEqual(early.status_code, 200, early.text)
        self.assertEqual(early.json()["row_count"], 11)

        late = self.post(plan(as_of_date=date(2026, 1, 15)))
        self.assertEqual(late.json()["row_count"], 12)

    def test_rows_never_exceed_the_registry_row_limit(self) -> None:
        body = self.post(plan()).json()
        limit = body["validation"]["row_limit"]
        self.assertLessEqual(body["row_count"], limit)

    def test_raw_database_errors_are_not_leaked(self) -> None:
        original = data_service.settings.database_path
        try:
            data_service.settings.database_path = ROOT / "data" / "curated" / "missing.duckdb"
            response = self.post(plan())
            self.assertEqual(response.status_code, 503)
            self.assertNotIn(".duckdb", response.json()["detail"])
        finally:
            data_service.settings.database_path = original


class ResultValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compiled = GuardedQueryCompiler(SchemaRegistry(REGISTRY)).compile(plan())
        self.controls = SchemaRegistry(REGISTRY).data["query_controls"]

    def good_row(self, **overrides) -> dict:
        row = {
            "obligor_id": "OBL-0008",
            "observation_date": "2025-01-31",
            "data_cutoff_date": "2025-02-05",
            "model_run_date": "2025-02-10",
            "portfolio": "corporate",
            "jurisdiction": "SAMA",
        }
        row = {**dict.fromkeys(self.compiled.selected_columns), **row}
        row.update(overrides)
        return row

    def test_clean_rows_pass(self) -> None:
        report = validate_result([self.good_row()], self.compiled, plan(), self.controls)
        self.assertTrue(report["passed"])

    def test_duplicate_grain_key_is_rejected(self) -> None:
        rows = [self.good_row(), self.good_row()]
        with self.assertRaises(ResultValidationError) as ctx:
            validate_result(rows, self.compiled, plan(), self.controls)
        self.assertIn("duplicate_grain_key", str(ctx.exception))

    def test_row_from_another_portfolio_is_rejected(self) -> None:
        with self.assertRaises(ResultValidationError) as ctx:
            validate_result(
                [self.good_row(portfolio="retail")], self.compiled, plan(), self.controls
            )
        self.assertIn("portfolio_mismatch", str(ctx.exception))

    def test_row_from_another_jurisdiction_is_rejected(self) -> None:
        with self.assertRaises(ResultValidationError) as ctx:
            validate_result(
                [self.good_row(jurisdiction="CBUAE")], self.compiled, plan(), self.controls
            )
        self.assertIn("jurisdiction_mismatch", str(ctx.exception))

    def test_row_published_after_the_as_of_date_is_rejected(self) -> None:
        # Defence in depth: even if the predicate were dropped, the returned data is
        # re-checked against the point-in-time bound.
        leaked = self.good_row(data_cutoff_date="2026-06-01")
        with self.assertRaises(ResultValidationError) as ctx:
            validate_result([leaked], self.compiled, plan(), self.controls)
        self.assertIn("point_in_time_breach:data_cutoff_date", str(ctx.exception))

    def test_observation_outside_the_requested_window_is_rejected(self) -> None:
        with self.assertRaises(ResultValidationError) as ctx:
            validate_result(
                [self.good_row(observation_date="2020-01-31")], self.compiled, plan(), self.controls
            )
        self.assertIn("observation_date_out_of_range", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
