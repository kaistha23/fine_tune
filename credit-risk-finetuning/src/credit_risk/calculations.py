from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from credit_risk.schemas import MetricValue


def _ratio(numerator: float | None, denominator: float | None, formula_id: str,
           sources: list[str], unit: str = "x") -> MetricValue:
    if numerator is None or denominator is None:
        return MetricValue(value=None, unit=unit, formula_id=formula_id, source_columns=sources,
                           missing_data_flag=True, validation_status="warning")
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return MetricValue(value=None, unit=unit, formula_id=formula_id, source_columns=sources,
                           validation_status="invalid")
    value = numerator / denominator
    if not math.isfinite(value):
        return MetricValue(value=None, unit=unit, formula_id=formula_id, source_columns=sources,
                           validation_status="invalid")
    return MetricValue(value=round(value, 6), unit=unit, formula_id=formula_id,
                       source_columns=sources)


def current_ratio(row: dict[str, Any]) -> MetricValue:
    return _ratio(row.get("current_assets"), row.get("current_liabilities"),
                  "ratio.current_ratio.v2", ["current_assets", "current_liabilities"])


def net_debt_to_ebitda(row: dict[str, Any]) -> MetricValue:
    debt, cash = row.get("total_debt"), row.get("cash")
    numerator = None if debt is None or cash is None else debt - cash
    return _ratio(numerator, row.get("ebitda"), "ratio.net_debt_to_ebitda.v2",
                  ["total_debt", "cash", "ebitda"])


def dscr(row: dict[str, Any]) -> MetricValue:
    return _ratio(row.get("operating_cash_flow"), row.get("debt_service"),
                  "ratio.dscr.v2", ["operating_cash_flow", "debt_service"])


def interest_coverage(row: dict[str, Any]) -> MetricValue:
    return _ratio(row.get("ebitda"), row.get("interest_expense"),
                  "ratio.interest_coverage.v2", ["ebitda", "interest_expense"])


def utilisation_pct(row: dict[str, Any]) -> MetricValue:
    result = _ratio(row.get("outstanding"), row.get("facility_limit"),
                    "ratio.utilisation.v2", ["outstanding", "facility_limit"], "pct")
    if isinstance(result.value, (int, float)):
        value = float(result.value) * 100
        result.value = round(value, 4) if math.isfinite(value) else None
        if result.value is None:
            result.validation_status = "invalid"
    return result


CALCULATORS: dict[str, Callable[[dict[str, Any]], MetricValue]] = {
    "current_ratio": current_ratio,
    "net_debt_to_ebitda": net_debt_to_ebitda,
    "dscr": dscr,
    "interest_coverage": interest_coverage,
    "utilisation_pct": utilisation_pct,
}

METRIC_UNITS = {
    "current_ratio": "x",
    "net_debt_to_ebitda": "x",
    "dscr": "x",
    "interest_coverage": "x",
    "utilisation_pct": "pct",
    "pit_pd": "probability",
    "internal_rating": "rating_grade",
    "stage": "stage",
    "days_past_due": "days",
}


def calculate_metrics(row: dict[str, Any], metrics: list[str]) -> dict[str, MetricValue]:
    return {name: CALCULATORS[name](row) for name in metrics if name in CALCULATORS}


def percentage_point_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return round((current - previous) * 100, 4)


def relative_change_pct(current: float | None, previous: float | None) -> float | None:
    if (current is None or previous is None or previous <= 0
            or not math.isfinite(current) or not math.isfinite(previous)):
        return None
    result = ((current - previous) / previous) * 100
    return round(result, 4) if math.isfinite(result) else None
