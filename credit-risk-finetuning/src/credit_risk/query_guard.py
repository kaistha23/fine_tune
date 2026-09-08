from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from credit_risk.schemas import EntityLevel, QueryPlan


class QueryGuardError(ValueError):
    pass


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    parameters: list[Any]
    source_table: str
    selected_columns: list[str]
    grain: str
    filters: list[str] = field(default_factory=list)
    joins: list[dict[str, Any]] = field(default_factory=list)
    governed_calculations: dict[str, str] = field(default_factory=dict)
    point_in_time_columns: list[str] = field(default_factory=list)


class SchemaRegistry:
    def __init__(self, path: str | Path, expected_version: str | None = None):
        with Path(path).open("r", encoding="utf-8") as handle:
            self.data = yaml.safe_load(handle)
        if self.data.get("default_deny") is not True:
            raise QueryGuardError("Schema registry must be default-deny")
        if expected_version and str(self.data.get("version")) != expected_version:
            raise QueryGuardError(
                f"Schema registry version {self.data.get('version')!r} does not match "
                f"required version {expected_version!r}"
            )

    @property
    def version(self) -> str:
        return str(self.data["version"])

    def validate_metric(self, metric: str, portfolio: str, grain: str) -> dict[str, Any]:
        config = self.data.get("metrics", {}).get(metric)
        if not config:
            raise QueryGuardError(f"Metric is not allowlisted: {metric}")
        if portfolio not in config.get("portfolios", []):
            raise QueryGuardError(f"Metric {metric} is not approved for {portfolio}")
        # Default-deny on grain. Checked before the column lookup so a reviewer is told the
        # metric is unavailable at this grain, not that the registry is missing columns.
        if grain not in config.get("entity_grain", []):
            raise QueryGuardError(
                f"Metric {metric} is not available at {grain} grain "
                f"(approved grains: {config.get('entity_grain', [])})"
            )
        return config


class GuardedQueryCompiler:
    def __init__(self, registry: SchemaRegistry):
        self.registry = registry

    def _assert_no_prohibited_keywords(self, sql: str) -> None:
        """Defence in depth on the SQL this compiler itself produced.

        Word-boundary matching matters: a plain substring test would flag
        "current_assets" for containing "set".
        """
        prohibited = self.registry.data["query_controls"].get("prohibited_sql_keywords", [])
        lowered = sql.lower()
        for keyword in prohibited:
            if re.search(r"\b" + re.escape(str(keyword).lower()) + r"\b", lowered):
                raise QueryGuardError(f"Generated SQL contains a prohibited keyword: {keyword}")

    def compile(self, plan: QueryPlan) -> CompiledQuery:
        controls = self.registry.data["query_controls"]
        if plan.jurisdiction.value not in controls["allowed_jurisdictions"]:
            raise QueryGuardError("Jurisdiction is not allowlisted")
        if len(plan.metrics) > controls["maximum_metrics"]:
            raise QueryGuardError("Too many requested metrics")

        month_span = (plan.date_to.year - plan.date_from.year) * 12 + (
            plan.date_to.month - plan.date_from.month
        ) + 1
        if month_span > controls["maximum_months"]:
            raise QueryGuardError("Requested date range exceeds the maximum history")

        table_name = (
            "obligor_monthly" if plan.entity_level == EntityLevel.OBLIGOR else "facility_monthly"
        )
        if table_name not in controls["allowed_tables"]:
            raise QueryGuardError(f"Table is not allowlisted: {table_name}")

        table = self.registry.data["tables"][table_name]
        allowed_columns = table["allowed_columns"]
        grain = table["grain"]

        pit = self.registry.data["point_in_time"]
        cutoff_column = pit["data_cutoff_column"]
        model_run_column = pit["model_run_column"]
        if cutoff_column not in allowed_columns:
            raise QueryGuardError(
                f"Table {table_name} does not declare a point-in-time cutoff column"
            )

        source_columns = {
            "obligor_id", "observation_date", "portfolio", "jurisdiction", cutoff_column
        }
        if plan.entity_level == EntityLevel.FACILITY:
            source_columns.add("facility_id")

        governed: dict[str, str] = {}
        uses_model_output = False
        for metric in plan.metrics:
            config = self.registry.validate_metric(metric, plan.portfolio.value, grain)
            metric_columns = list(config.get("dependencies", []))
            if config.get("source_column"):
                metric_columns.append(config["source_column"])
            source_columns.update(metric_columns)
            if config.get("formula_id"):
                governed[metric] = config["formula_id"]
            if any(allowed_columns.get(c, {}).get("model_output") for c in metric_columns):
                uses_model_output = True

        # Model outputs carry their own asserted-as-of date. A row can sit inside the data
        # cutoff while its PD/LGD/ECL were produced by a later model run.
        if uses_model_output:
            if model_run_column not in allowed_columns:
                raise QueryGuardError(
                    f"Table {table_name} exposes model outputs but declares no "
                    f"{model_run_column} column"
                )
            source_columns.add(model_run_column)

        unknown = source_columns.difference(allowed_columns)
        if unknown:
            raise QueryGuardError(f"Schema registry is missing approved columns: {sorted(unknown)}")

        ordered = sorted(source_columns)
        quoted = ", ".join('"' + column + '"' for column in ordered)

        predicates = [
            '"obligor_id" = ?', '"portfolio" = ?', '"jurisdiction" = ?',
            '"observation_date" BETWEEN ? AND ?',
        ]
        parameters: list[Any] = [
            plan.obligor_id, plan.portfolio.value, plan.jurisdiction.value,
            plan.date_from, plan.date_to,
        ]
        filters = [
            "obligor_id = <masked>",
            f"portfolio = {plan.portfolio.value}",
            f"jurisdiction = {plan.jurisdiction.value}",
            f"observation_date BETWEEN {plan.date_from} AND {plan.date_to}",
        ]

        if plan.entity_level == EntityLevel.FACILITY:
            predicates.append('"facility_id" = ?')
            parameters.append(plan.facility_id)
            filters.append("facility_id = <masked>")

        # Mandatory point-in-time predicates. Not optional, not caller-supplied.
        pit_columns = [cutoff_column]
        predicates.append('"' + cutoff_column + '" <= ?')
        parameters.append(plan.as_of_date)
        filters.append(f"{cutoff_column} <= {plan.as_of_date} (point-in-time)")
        if uses_model_output:
            pit_columns.append(model_run_column)
            predicates.append('"' + model_run_column + '" <= ?')
            parameters.append(plan.as_of_date)
            filters.append(f"{model_run_column} <= {plan.as_of_date} (point-in-time)")

        maximum_rows = int(controls["maximum_result_rows"])
        sql = (
            f'SELECT {quoted} FROM "{table_name}" WHERE '
            + " AND ".join(predicates)
            + f' ORDER BY "observation_date" ASC LIMIT {maximum_rows}'
        )
        self._assert_no_prohibited_keywords(sql)

        return CompiledQuery(
            sql=sql,
            parameters=parameters,
            source_table=table_name,
            selected_columns=ordered,
            grain=grain,
            filters=filters,
            joins=[],
            governed_calculations=governed,
            point_in_time_columns=pit_columns,
        )
