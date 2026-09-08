"""Turn validated rows into the compact factsheet the model actually sees (finding H7).

calculations.py was correct and unit-tested but imported by nothing, and CreditFactsheet
was never constructed, so the pipeline stopped at raw rows. This is the missing step: it
is where deterministic Python owns every number before an LLM is allowed near it.

Both plans are explicit that hundreds of raw columns must not be pasted into a prompt.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from credit_risk.calculations import (
    CALCULATORS, calculate_metrics, percentage_point_change, relative_change_pct,
)
from credit_risk.schemas import CreditFactsheet, MetricValue, QueryPlan

# Columns reported as the obligor's position now, when present at the queried grain.
POSITION_COLUMNS = (
    "outstanding", "facility_limit", "days_past_due", "internal_rating", "stage",
)
# Risk-model outputs are reported, never recomputed. The LLM must not derive these.
MODEL_OUTPUT_COLUMNS = ("ttc_pd", "pit_pd", "lgd", "ead", "ecl")

# Changes in a probability are reported in percentage points, not as a relative change:
# 2% to 5% is +3pp, not +150%. Both plans call this out as a known failure mode.
PROBABILITY_COLUMNS = ("ttc_pd", "pit_pd", "lgd")


class FactsheetError(ValueError):
    pass


def _observation_date(row: dict[str, Any]) -> date | None:
    value = row.get("observation_date")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (_observation_date(row) or date.min))


def build_trends(first: dict[str, Any], last: dict[str, Any],
                 metrics: dict[str, MetricValue]) -> dict[str, Any]:
    """Change across the observed window, using the right units for each quantity."""
    trends: dict[str, Any] = {}
    for column in PROBABILITY_COLUMNS:
        if first.get(column) is not None and last.get(column) is not None:
            trends[f"{column}_change_pp"] = percentage_point_change(
                last[column], first[column])
    for column in ("revenue", "ebitda", "outstanding"):
        if first.get(column) is not None and last.get(column) is not None:
            trends[f"{column}_change_pct"] = relative_change_pct(last[column], first[column])
    if "utilisation_pct" in metrics and metrics["utilisation_pct"].value is not None:
        opening = CALCULATORS["utilisation_pct"](first)
        if opening.value is not None:
            trends["utilisation_change_pp"] = round(
                float(metrics["utilisation_pct"].value) - float(opening.value), 4)
    return trends


def detect_events(rows: list[dict[str, Any]]) -> list[str]:
    events: list[str] = []
    dpd_30 = sum(1 for row in rows if (row.get("days_past_due") or 0) >= 30)
    if dpd_30:
        events.append(f"{dpd_30}_months_at_30dpd_or_worse")
    if any(row.get("stage") == 3 for row in rows):
        events.append("stage_3_observed_in_window")
    elif any(row.get("stage") == 2 for row in rows):
        events.append("stage_2_observed_in_window")
    if any(row.get("restructuring_flag") for row in rows):
        events.append("restructuring_flagged")
    if any(row.get("watchlist_flag") for row in rows):
        events.append("watchlist_flagged")
    stages = [row.get("stage") for row in rows if row.get("stage") is not None]
    if len(stages) >= 2 and stages[-1] > stages[0]:
        events.append(f"stage_migration_{stages[0]}_to_{stages[-1]}")
    return events


def detect_data_quality(rows: list[dict[str, Any]], plan: QueryPlan) -> list[str]:
    flags: list[str] = []
    dates = [d for d in (_observation_date(row) for row in rows) if d is not None]
    if len(dates) != len(rows):
        flags.append("unparseable_observation_date")
    if len(set(dates)) != len(dates):
        flags.append("duplicate_observation_dates")
    if dates:
        latest = max(dates)
        stale_months = (plan.as_of_date.year - latest.year) * 12 + (
            plan.as_of_date.month - latest.month)
        if stale_months > 3:
            flags.append(f"latest_observation_{stale_months}_months_old")
    return flags


def build_factsheet(rows: list[dict[str, Any]], plan: QueryPlan,
                    case_id: str | None = None) -> CreditFactsheet:
    """Build a factsheet from rows the data service has already validated."""
    if not rows:
        raise FactsheetError("Cannot build a factsheet from an empty result set")

    ordered = _sorted_rows(rows)
    first, last = ordered[0], ordered[-1]

    metrics = calculate_metrics(last, plan.metrics)
    # Metrics sourced straight from a governed column are reported, not recalculated.
    for metric in plan.metrics:
        if metric not in metrics and metric in last:
            metrics[metric] = MetricValue(
                value=last[metric], source_columns=[metric], formula_id=None,
                missing_data_flag=last[metric] is None,
                validation_status="warning" if last[metric] is None else "valid",
            )

    position = {c: last[c] for c in POSITION_COLUMNS if c in last}
    model_outputs = {c: last[c] for c in MODEL_OUTPUT_COLUMNS if c in last}

    missing = sorted(
        {column for column, value in last.items() if value is None}
        | {name for name, metric in metrics.items() if metric.missing_data_flag}
    )

    return CreditFactsheet(
        case_id=case_id or f"CASE-{plan.obligor_id}-{plan.as_of_date.isoformat()}",
        obligor_id=plan.obligor_id,
        portfolio=plan.portfolio,
        jurisdiction=plan.jurisdiction,
        as_of_date=plan.as_of_date,
        observation_months=len(ordered),
        current_position=position,
        calculated_metrics=metrics,
        trends=build_trends(first, last, metrics),
        events=detect_events(ordered),
        model_outputs=model_outputs,
        missing_information=missing,
        data_quality_flags=detect_data_quality(ordered, plan),
    )
