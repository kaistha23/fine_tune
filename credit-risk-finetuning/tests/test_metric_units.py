from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from credit_risk.calculations import CALCULATORS, METRIC_UNITS
from credit_risk.schemas import MetricValue


ROOT = Path(__file__).resolve().parents[1]
ROW = {
    "current_assets": 2.0,
    "current_liabilities": 1.0,
    "total_debt": 5.0,
    "cash": 1.0,
    "ebitda": 2.0,
    "operating_cash_flow": 2.0,
    "debt_service": 1.0,
    "interest_expense": 1.0,
    "outstanding": 80.0,
    "facility_limit": 100.0,
}


@pytest.mark.parametrize("metric", sorted(CALCULATORS))
def test_calculated_metric_always_carries_its_governed_unit(metric):
    assert CALCULATORS[metric](ROW).unit == METRIC_UNITS[metric]


def test_metric_value_rejects_a_missing_unit():
    with pytest.raises(ValidationError):
        MetricValue(value=1.0)


def test_registry_units_match_the_factsheet_unit_contract():
    registry = yaml.safe_load((ROOT / "configs/schema_registry.yaml").read_text())
    assert registry["version"] == "1.5.0"
    assert {name: spec["unit"] for name, spec in registry["metrics"].items()} == METRIC_UNITS
