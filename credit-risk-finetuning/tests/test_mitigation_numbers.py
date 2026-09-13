import math

import pytest

from credit_risk.calculations import CALCULATORS, relative_change_pct

RATIOS = [("current_ratio", "current_assets", "current_liabilities"),
          ("net_debt_to_ebitda", "total_debt", "ebitda"),
          ("dscr", "operating_cash_flow", "debt_service"),
          ("interest_coverage", "ebitda", "interest_expense"),
          ("utilisation_pct", "outstanding", "facility_limit")]


@pytest.mark.parametrize("name,numerator,denominator", RATIOS)
@pytest.mark.parametrize("value", [0, -30, float("nan"), float("inf"), -float("inf")])
def test_invalid_denominators(name, numerator, denominator, value):
    result = CALCULATORS[name]({numerator: 90, denominator: value, "cash": 0})
    assert result.value is None and result.validation_status == "invalid"
    assert result.formula_id.endswith(".v2")


@pytest.mark.parametrize("name,numerator,denominator", RATIOS)
def test_missing_tiny_positive_and_overflow(name, numerator, denominator):
    assert CALCULATORS[name]({}).missing_data_flag
    result = CALCULATORS[name]({numerator: 1, denominator: 1e-12, "cash": 0})
    assert result.validation_status == "valid" and math.isfinite(result.value)
    for values in [{numerator: 1e308, denominator: 1e-308, "cash": 0},
                   {numerator: float("nan"), denominator: 1, "cash": 0}]:
        result = CALCULATORS[name](values)
        assert result.validation_status == "invalid" and result.value is None


def test_negative_numerators_remain_meaningful():
    assert CALCULATORS["net_debt_to_ebitda"]({"total_debt": 10, "cash": 30, "ebitda": 10}).value == -2
    assert CALCULATORS["interest_coverage"]({"ebitda": -30, "interest_expense": 10}).value == -3


@pytest.mark.parametrize("previous", [-10, 0, None, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_relative_base(previous):
    assert relative_change_pct(20, previous) is None


def test_invalid_metric_reaches_factsheet_quality_flags():
    from test_factsheet import rows, plan
    from credit_risk.factsheet import build_factsheet
    inputs = rows()
    inputs[-1]["current_liabilities"] = -30
    sheet = build_factsheet(inputs, plan())
    assert sheet.calculated_metrics["current_ratio"].value is None
    assert "invalid_metric:current_ratio" in sheet.data_quality_flags
