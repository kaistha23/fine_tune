from __future__ import annotations

from dataclasses import dataclass
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

    def validate_metric(self, metric: str, portfolio: str) -> dict[str, Any]:
        config = self.data.get("metrics", {}).get(metric)
        if not config:
            raise QueryGuardError(f"Metric is not allowlisted: {metric}")
        if portfolio not in config.get("portfolios", []):
            raise QueryGuardError(f"Metric {metric} is not approved for {portfolio}")
        return config


class GuardedQueryCompiler:
    def __init__(self, registry: SchemaRegistry):
        self.registry = registry

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
        table = self.registry.data["tables"][table_name]
        allowed_columns = table["allowed_columns"]
        source_columns = {"obligor_id", "observation_date", "portfolio", "jurisdiction"}
        if plan.entity_level == EntityLevel.FACILITY:
            source_columns.add("facility_id")

        for metric in plan.metrics:
            config = self.registry.validate_metric(metric, plan.portfolio.value)
            dependencies = config.get("dependencies", [])
            source = config.get("source_column")
            source_columns.update(dependencies)
            if source:
                source_columns.add(source)

        unknown = source_columns.difference(allowed_columns)
        if unknown:
            raise QueryGuardError(f"Schema registry is missing approved columns: {sorted(unknown)}")

        ordered = sorted(source_columns)
        quoted = ", ".join(f'"{column}"' for column in ordered)
        predicates = [
            '"obligor_id" = ?', '"portfolio" = ?', '"jurisdiction" = ?',
            '"observation_date" BETWEEN ? AND ?'
        ]
        parameters: list[Any] = [
            plan.obligor_id, plan.portfolio.value, plan.jurisdiction.value,
            plan.date_from, plan.date_to
        ]
        if plan.entity_level == EntityLevel.FACILITY:
            predicates.append('"facility_id" = ?')
            parameters.append(plan.facility_id)

        maximum_rows = int(controls["maximum_result_rows"])
        sql = (
            f'SELECT {quoted} FROM "{table_name}" WHERE '
            + " AND ".join(predicates)
            + f' ORDER BY "observation_date" ASC LIMIT {maximum_rows}'
        )
        return CompiledQuery(sql=sql, parameters=parameters, source_table=table_name,
                             selected_columns=ordered)

