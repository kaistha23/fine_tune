from __future__ import annotations

from collections.abc import Callable
from typing import Any

from credit_risk.schemas import MetricValue


def _ratio(numerator: float | None, denominator: float | None, formula_id: str,
           sources: list[str]) -> MetricValue:
    if numerator is None or denominator is None:
        return MetricValue(value=None, formula_id=formula_id, source_columns=sources,
                           missing_data_flag=True, validation_status="warning")
    if denominator == 0:
        return MetricValue(value=None, formula_id=formula_id, source_columns=sources,
                           validation_status="invalid")
    return MetricValue(value=round(numerator / denominator, 6), formula_id=formula_id,
                       source_columns=sources)


def current_ratio(row: dict[str, Any]) -> MetricValue:
    return _ratio(row.get("current_assets"), row.get("current_liabilities"),
                  "ratio.current_ratio.v1", ["current_assets", "current_liabilities"])


def net_debt_to_ebitda(row: dict[str, Any]) -> MetricValue:
    debt, cash = row.get("total_debt"), row.get("cash")
    numerator = None if debt is None or cash is None else debt - cash
    return _ratio(numerator, row.get("ebitda"), "ratio.net_debt_to_ebitda.v1",
                  ["total_debt", "cash", "ebitda"])


def dscr(row: dict[str, Any]) -> MetricValue:
    return _ratio(row.get("operating_cash_flow"), row.get("debt_service"),
                  "ratio.dscr.v1", ["operating_cash_flow", "debt_service"])


def interest_coverage(row: dict[str, Any]) -> MetricValue:
    return _ratio(row.get("ebitda"), row.get("interest_expense"),
                  "ratio.interest_coverage.v1", ["ebitda", "interest_expense"])


def utilisation_pct(row: dict[str, Any]) -> MetricValue:
    result = _ratio(row.get("outstanding"), row.get("facility_limit"),
                    "ratio.utilisation.v1", ["outstanding", "facility_limit"])
    if isinstance(result.value, (int, float)):
        result.value = round(float(result.value) * 100, 4)
    return result


CALCULATORS: dict[str, Callable[[dict[str, Any]], MetricValue]] = {
    "current_ratio": current_ratio,
    "net_debt_to_ebitda": net_debt_to_ebitda,
    "dscr": dscr,
    "interest_coverage": interest_coverage,
    "utilisation_pct": utilisation_pct,
}


def calculate_metrics(row: dict[str, Any], metrics: list[str]) -> dict[str, MetricValue]:
    return {name: CALCULATORS[name](row) for name in metrics if name in CALCULATORS}


def percentage_point_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return round((current - previous) * 100, 4)


def relative_change_pct(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return round(((current - previous) / previous) * 100, 4)
