from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from credit_risk.schemas import EntityLevel, QueryPlan


class QueryGuardError(ValueError):
    pass


_SYMBOL_OPERATORS = ("<=", ">=", "<>", "!=", "=", "<", ">")
_WORD_OPERATORS = ("BETWEEN", "LIKE", "ILIKE", "IN", "OR", "AND", "NOT")
_OPERATOR_PATTERN = re.compile("|".join(
    list(_SYMBOL_OPERATORS)
    + ["(?<![A-Z_])" + word + "(?![A-Z_])" for word in _WORD_OPERATORS]
))


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
    # Cohort queries only. The reviewer sees the dimensions, the aggregation applied to
    # each metric, and the cohort-size floor the HAVING clause enforces.
    group_by: list[str] = field(default_factory=list)
    aggregations: dict[str, str] = field(default_factory=dict)
    minimum_cohort_size: int | None = None


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

    def _assert_operators_allowlisted(self, sql: str) -> None:
        """Every operator in the generated SQL must be declared in the registry.

        Word operators use lookarounds so ORDER BY does not read as OR and
        internal_rating does not read as IN.
        """
        allowed = {
            str(op).upper()
            for op in self.registry.data["query_controls"].get("allowed_operators", [])
        }
        found = set(_OPERATOR_PATTERN.findall(sql.upper()))
        undeclared = sorted(found - allowed)
        if undeclared:
            raise QueryGuardError(f"Generated SQL uses undeclared operators: {undeclared}")

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

        # A portfolio cohort is a cohort of obligors, so it reads the obligor table and
        # aggregates it. Facility-grain cohorts would double-count borrowers with several
        # facilities, which is exactly the grain mismatch the plans call out.
        table_name = (
            "facility_monthly" if plan.entity_level == EntityLevel.FACILITY
            else "obligor_monthly"
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

        is_cohort = plan.entity_level == EntityLevel.PORTFOLIO

        # Columns the query touches, checked against the allowlist. The projection is
        # built per branch below: a cohort must not select obligor_id at all.
        source_columns = {"observation_date", "portfolio", "jurisdiction", cutoff_column}
        if not is_cohort:
            source_columns.add("obligor_id")
        if plan.entity_level == EntityLevel.FACILITY:
            source_columns.add("facility_id")

        cohort = controls.get("cohort") or {}
        if is_cohort and not cohort:
            raise QueryGuardError("Schema registry declares no cohort controls")

        group_by: list[str] = []
        if is_cohort:
            allowed_group_by = list(cohort.get("allowed_group_by", []))
            maximum_group_by = int(cohort.get("maximum_group_by", 0))
            if len(plan.group_by) > maximum_group_by:
                raise QueryGuardError(
                    f"Too many cohort dimensions: {len(plan.group_by)} > {maximum_group_by}"
                )
            for dimension in plan.group_by:
                if dimension not in allowed_group_by:
                    raise QueryGuardError(
                        f"Cohort dimension is not allowlisted: {dimension}")
                if allowed_columns.get(dimension, {}).get("sensitivity") == "restricted":
                    # Belt and braces: the allowlist above already excludes identifiers,
                    # but grouping on a restricted column is a per-entity extract however
                    # it got into the list.
                    raise QueryGuardError(
                        f"Cohort dimension is a restricted column: {dimension}")
                source_columns.add(dimension)
            # portfolio and jurisdiction are always grouped, so every returned row carries
            # them and the consistency checks in the data service still have something to
            # assert against.
            group_by = ["portfolio", "jurisdiction", *plan.group_by]

        governed: dict[str, str] = {}
        aggregations: dict[str, str] = {}
        uses_model_output = False
        aggregatable_types = set(cohort.get("aggregatable_types", []))
        allowed_aggregations = {str(a).lower() for a in cohort.get("allowed_aggregations", [])}

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

            if not is_cohort:
                continue

            # A governed ratio cannot be aggregated in SQL. AVG(numerator)/AVG(denominator)
            # is not AVG(ratio), and for something like DSCR the two differ by enough to
            # change a credit decision. Ratios have to be computed per obligor and then
            # aggregated, which this single-pass compiler does not do.
            if config.get("formula_id"):
                raise QueryGuardError(
                    f"Metric {metric} is a governed calculation ({config['formula_id']}) "
                    "and cannot be aggregated in SQL: the average of a ratio is not the "
                    "ratio of averages. Request its stored dependencies instead."
                )
            column = config.get("source_column")
            if not column:
                raise QueryGuardError(
                    f"Metric {metric} has no stored column to aggregate")
            aggregation = plan.cohort_aggregation.lower()
            if aggregation not in allowed_aggregations:
                raise QueryGuardError(
                    f"Aggregation is not allowlisted: {aggregation}")
            column_type = allowed_columns.get(column, {}).get("type")
            if aggregation in ("sum", "avg") and column_type not in aggregatable_types:
                raise QueryGuardError(
                    f"Cannot apply {aggregation} to {metric}: column {column} is "
                    f"{column_type}, and only {sorted(aggregatable_types)} are aggregatable"
                )
            aggregations[metric] = f"{aggregation}({column})"

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

        predicates = ['"portfolio" = ?', '"jurisdiction" = ?',
                      '"observation_date" BETWEEN ? AND ?']
        parameters: list[Any] = [
            plan.portfolio.value, plan.jurisdiction.value, plan.date_from, plan.date_to,
        ]
        filters = [
            f"portfolio = {plan.portfolio.value}",
            f"jurisdiction = {plan.jurisdiction.value}",
            f"observation_date BETWEEN {plan.date_from} AND {plan.date_to}",
        ]

        if not is_cohort:
            predicates.insert(0, '"obligor_id" = ?')
            parameters.insert(0, plan.obligor_id)
            filters.insert(0, "obligor_id = <masked>")
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
        # One row over the limit, so a truncated result is detectable rather than silently
        # returned as if it were complete. The data service rejects the overflow row.
        fetch_limit = maximum_rows + 1

        minimum_cohort_size: int | None = None
        if is_cohort:
            minimum_cohort_size = int(cohort["minimum_cohort_size"])
            entity_column = table["entity_column"]
            # COUNT(DISTINCT obligor_id), not COUNT(*). The table is obligor-month grain,
            # so a group spanning twelve months reaches COUNT(*) = 25 with three
            # borrowers in it. Counting rows would make the floor look like k-anonymity
            # while providing none. The identifier appears only inside the aggregate: it
            # is never projected, never filtered on and never grouped by.
            cohort_count = f'COUNT(DISTINCT "{entity_column}")'
            projection = [f'"{column}"' for column in group_by]
            # cohort_size is always projected: it is what lets the data service re-assert
            # the floor on the rows that actually came back.
            projection.append(f"{cohort_count} AS cohort_size")
            for metric, expression in sorted(aggregations.items()):
                function, _, column = expression.partition("(")
                projection.append(
                    f'{function.upper()}("{column.rstrip(")")}") AS "{metric}"')
            ordered = [*group_by, "cohort_size", *sorted(aggregations)]
            grouped = ", ".join(f'"{column}"' for column in group_by)
            # Order on a dimension that actually varies. portfolio and jurisdiction are
            # constant within a cohort result, so ordering on them is no ordering at all.
            order_column = (
                "observation_date" if "observation_date" in plan.group_by
                else (plan.group_by[0] if plan.group_by else group_by[0])
            )
            sql = (
                f'SELECT {", ".join(projection)} FROM "{table_name}" WHERE '
                + " AND ".join(predicates)
                + f" GROUP BY {grouped}"
                + f" HAVING {cohort_count} >= ?"
                + f' ORDER BY "{order_column}" ASC LIMIT {fetch_limit}'
            )
            parameters.append(minimum_cohort_size)
            filters.append(f"cohort_size >= {minimum_cohort_size} (small cells suppressed)")
            result_grain = "_".join(group_by)
        else:
            ordered = sorted(source_columns)
            quoted = ", ".join('"' + column + '"' for column in ordered)
            sql = (
                f'SELECT {quoted} FROM "{table_name}" WHERE '
                + " AND ".join(predicates)
                + f' ORDER BY "observation_date" ASC LIMIT {fetch_limit}'
            )
            result_grain = grain

        self._assert_no_prohibited_keywords(sql)
        self._assert_operators_allowlisted(sql)

        return CompiledQuery(
            sql=sql,
            parameters=parameters,
            source_table=table_name,
            selected_columns=ordered,
            grain=result_grain,
            filters=filters,
            joins=[],
            governed_calculations=governed,
            point_in_time_columns=pit_columns,
            group_by=group_by,
            aggregations=aggregations,
            minimum_cohort_size=minimum_cohort_size,
        )
