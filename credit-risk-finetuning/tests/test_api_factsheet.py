"""The /v1/factsheet route: approved rows in, compact factsheet out (finding H7).

factsheet.py was tested but had no caller in src/, so the calculation layer still was not
reachable from the running system.
"""

import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path

from http_helpers import TestClient, execute_reviewed

from credit_risk import api, data_service
from credit_risk.schemas import Jurisdiction, Portfolio, QueryPlan

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "data" / "curated" / "credit_risk.duckdb"


def ensure_fixture() -> None:
    if not FIXTURE.is_file():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_fixture.py")],
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
        "metrics": ["current_ratio", "dscr", "utilisation_pct", "pit_pd", "stage"],
    }
    base.update(overrides)
    return QueryPlan(**base)


class FactsheetRouteTests(unittest.TestCase):
    """The API calls the data service over HTTP, so the transport is stubbed and the
    data service app is invoked directly - which is what the internal network does."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_fixture()
        cls._role = data_service.settings.service_role

    @classmethod
    def tearDownClass(cls) -> None:
        data_service.settings.service_role = cls._role

    def setUp(self) -> None:
        role = data_service.settings.service_role
        data_service.settings.service_role = "data_service"
        ds_client = TestClient(data_service.app)
        data_service.settings.service_role = role

        def fake_rows(request):
            saved = data_service.settings.service_role
            data_service.settings.service_role = "data_service"
            try:
                response = execute_reviewed(ds_client, request.query_plan.model_dump(mode="json"))
                return response.json()
            finally:
                data_service.settings.service_role = saved

        self._original = api._request_rows
        api._request_rows = fake_rows
        self.client = TestClient(api.app)

    def tearDown(self) -> None:
        api._request_rows = self._original

    def test_factsheet_route_returns_a_compact_factsheet(self) -> None:
        response = self.client.post(
            "/v1/factsheet",
            json={"user_text": "Analyse the obligor", "query_plan": plan().model_dump(mode="json")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        sheet = body["factsheet"]
        self.assertEqual(body["row_count"], 12)
        self.assertEqual(sheet["schema_version"], "credit_factsheet_v1")
        self.assertEqual(sheet["obligor_id"], "OBL-0008")
        self.assertEqual(sheet["observation_months"], 12)
        self.assertIn("current_ratio", sheet["calculated_metrics"])
        self.assertEqual(
            sheet["calculated_metrics"]["current_ratio"]["formula_id"], "ratio.current_ratio.v2"
        )

    def test_factsheet_is_smaller_than_the_rows_it_came_from(self) -> None:
        # The whole point: the model sees a factsheet, never 12 months of raw columns.
        response = self.client.post(
            "/v1/factsheet",
            json={"user_text": "Analyse the obligor", "query_plan": plan().model_dump(mode="json")},
        )
        self.assertNotIn("rows", response.json())


if __name__ == "__main__":
    unittest.main()
