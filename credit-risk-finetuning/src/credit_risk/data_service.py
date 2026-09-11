from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError

from credit_risk.architecture_policy import ArchitecturePolicy, ArchitecturePolicyError
from credit_risk.query_guard import (
    CompiledQuery,
    GuardedQueryCompiler,
    QueryGuardError,
    SchemaRegistry,
)
from credit_risk.review_store import ReviewConflict, ReviewStore, canonical, digest
from credit_risk.schemas import EntityLevel, QueryPlan
from credit_risk.settings import settings

app = FastAPI(title="Restricted Credit Data Service", version="0.1.0", docs_url=None)
policy = ArchitecturePolicy(settings.architecture_policy, settings.architecture_policy_version)
registry = SchemaRegistry(settings.schema_registry, settings.schema_registry_version)
compiler = GuardedQueryCompiler(registry)


def authorize_service(x_service_token: str = Header(default="")) -> None:
    # Constant-time comparison so the token cannot be recovered by timing (M9).
    if not settings.service_token or not secrets.compare_digest(
        x_service_token, settings.service_token
    ):
        raise HTTPException(status_code=401, detail="Invalid internal service credential")


class InternalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer_id: str
    query_plan: QueryPlan
    revision_id: str | None = None


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer_id: str
    revision_id: str
    decision: Literal["approved", "rejected"]
    comment: str = ""
    corrected_query_plan: dict | None = None


def store():
    return ReviewStore(settings.review_store_path)


def snapshot_hash():
    if not settings.database_path.is_file():
        raise HTTPException(503, "Analytical snapshot unavailable")
    with settings.database_path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def binding(plan, compiled, snapshot):
    # Keyed digest prevents guessing sensitive parameters from the review packet.
    content = canonical(
        {
            "plan": plan.model_dump(mode="json"),
            "sql": compiled.sql,
            "parameters": compiled.parameters,
            "snapshot": snapshot,
            "schema": registry.data,
            "policy": policy.data,
        }
    )
    return hmac.new(settings.service_token.encode(), content.encode(), hashlib.sha256).hexdigest()


def build_sql_review_packet(compiled):
    return {
        "query_hash": digest({"schema": registry.version, "sql": compiled.sql}),
        "parameterised_sql": compiled.sql,
        "parameter_placeholders": len(compiled.parameters),
        "parameter_values": ["***MASKED***"] * len(compiled.parameters),
        "source_table": compiled.source_table,
        "tables": [compiled.source_table],
        "selected_columns": compiled.selected_columns,
        "grain": compiled.grain,
        "joins": compiled.joins,
        "filters": compiled.filters,
        "governed_calculations": compiled.governed_calculations,
        "point_in_time_columns": compiled.point_in_time_columns,
        "group_by": compiled.group_by,
        "aggregations": compiled.aggregations,
        "minimum_cohort_size": compiled.minimum_cohort_size,
        "schema_registry_version": registry.version,
        "architecture_policy_version": policy.version,
        "editable": False,
        "execution_status": "NOT_EXECUTED",
    }


def prepare(plan, reviewer, parent=None):
    policy.require(settings.service_role, "compile_parameterised_sql")
    compiled = compiler.compile(plan)
    snapshot = snapshot_hash()
    fingerprint = binding(plan, compiled, snapshot)
    packet = {
        **build_sql_review_packet(compiled),
        "request_digest": fingerprint,
        "snapshot_sha256": snapshot,
    }
    return store().create(reviewer, fingerprint, plan.model_dump(mode="json"), packet, parent)


@app.post("/internal/v1/query/prepare", dependencies=[Depends(authorize_service)])
def prepare_query(request: InternalRequest):
    try:
        return prepare(request.query_plan, request.reviewer_id)
    except (QueryGuardError, ArchitecturePolicyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/internal/v1/query/review", dependencies=[Depends(authorize_service)])
def decide_query(request: DecisionRequest):
    if (
        request.decision == "rejected" or request.corrected_query_plan is not None
    ) and not request.comment.strip():
        raise HTTPException(422, "Rejection requires a comment")
    try:
        original = store().get(request.revision_id, request.reviewer_id)
        if request.decision == "approved" and request.corrected_query_plan is None:
            plan = QueryPlan.model_validate_json(original["plan"])
            if binding(plan, compiler.compile(plan), snapshot_hash()) != original["binding"]:
                raise ReviewConflict("Snapshot or controls changed; prepare again")
        store().decide(
            request.revision_id,
            request.reviewer_id,
            request.decision,
            request.comment,
            request.corrected_query_plan,
        )
    except ReviewConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    corrected = None
    if request.corrected_query_plan is not None:
        try:
            plan = QueryPlan.model_validate(request.corrected_query_plan)
            corrected = prepare(plan, request.reviewer_id, request.revision_id)
            store().validated_correction(request.revision_id, request.reviewer_id, plan, corrected)
        except (ValidationError, QueryGuardError, ArchitecturePolicyError) as exc:
            store().failed_correction(request.revision_id, request.reviewer_id, type(exc).__name__)
            raise HTTPException(
                422, "Corrected query plan failed revalidation; original rejected"
            ) from exc
    return {
        "sql_review_status": "rejected"
        if request.corrected_query_plan is not None
        else request.decision,
        "revision_id": request.revision_id,
        "corrected_sql_review": corrected,
        "revalidated": corrected is not None,
        "eligible_for_training": False,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _as_date(value: Any) -> date | None:
    """Coerce a cell to a date for validation.

    Rows reaching validate_result have already been through _json_safe, so dates arrive
    as ISO strings. Without the string branch every date check below silently passes.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


class ResultValidationError(ValueError):
    """Raised when returned rows do not match the grain and filters that were approved."""


def validate_result(
    rows: list[dict[str, Any]], compiled: CompiledQuery, plan: QueryPlan, controls: dict[str, Any]
) -> dict[str, Any]:
    """Check the rows actually returned against the contract the plan was approved under.

    Every check here is defence in depth: the compiled SQL should already guarantee it.
    A failure means the data does not match its declared schema, which is exactly the
    condition a factsheet must never be built on.
    """
    failures: list[str] = []

    maximum_rows = int(controls["maximum_result_rows"])
    # The compiler asks for one row more than the limit, so an over-long result means the
    # answer was truncated. A partial cohort table, or a time series missing its tail, is
    # worse than no answer: it looks complete and is quietly wrong.
    if len(rows) > maximum_rows:
        failures.append("result_truncated")

    if plan.entity_level == EntityLevel.PORTFOLIO:
        grain_keys = list(compiled.group_by)
    elif plan.entity_level == EntityLevel.OBLIGOR:
        grain_keys = ["obligor_id", "observation_date"]
    else:
        grain_keys = ["obligor_id", "facility_id", "observation_date"]
    definitions = registry.data["tables"][compiled.source_table]["allowed_columns"]
    for row in rows:
        if set(row) != set(compiled.selected_columns):
            failures.append("result_columns_mismatch")
        if plan.entity_level != EntityLevel.PORTFOLIO:
            if row.get("obligor_id") != plan.obligor_id:
                failures.append("obligor_mismatch")
            if (
                plan.entity_level == EntityLevel.FACILITY
                and row.get("facility_id") != plan.facility_id
            ):
                failures.append("facility_mismatch")
        required = set(grain_keys)
        if plan.entity_level != EntityLevel.PORTFOLIO:
            required.update(compiled.point_in_time_columns)
        for column in compiled.selected_columns:
            value = row.get(column)
            rule = definitions.get(column, {})
            if value is None:
                if column in required:
                    failures.append("missing_required_field")
                continue
            kind = rule.get("type")
            if column in compiled.aggregations:
                kind = "number"
            valid = True
            if kind in ("number", "probability", "integer"):
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                )
                if kind == "integer":
                    valid = valid and isinstance(value, int)
                if valid and column not in compiled.aggregations:
                    valid = ("min" not in rule or value >= rule["min"]) and (
                        "max" not in rule or value <= rule["max"]
                    )
            elif kind == "string":
                valid = isinstance(value, str) and bool(value.strip())
            elif kind == "boolean":
                valid = isinstance(value, bool)
            elif kind == "date":
                try:
                    date.fromisoformat(value) if isinstance(value, str) else date.fromisoformat(
                        value.isoformat()
                    )
                except (ValueError, TypeError, AttributeError):
                    valid = False
            if "allowed" in rule and column not in compiled.aggregations:
                valid = valid and value in rule["allowed"]
            if not valid:
                failures.append("invalid_field:" + column)
    seen: set[tuple] = set()
    for row in rows:
        key = tuple(row.get(column) for column in grain_keys)
        if key in seen:
            failures.append("duplicate_grain_key")
            break
        seen.add(key)

    for row in rows:
        if row.get("portfolio") != plan.portfolio.value:
            failures.append("portfolio_mismatch")
            break
    for row in rows:
        if row.get("jurisdiction") != plan.jurisdiction.value:
            failures.append("jurisdiction_mismatch")
            break

    # Re-assert the cohort floor on the rows that came back, not just the HAVING clause
    # that asked for it. A cohort below the floor is an individual disclosure, so this is
    # the one check that must not depend on the SQL having been compiled correctly.
    if compiled.minimum_cohort_size is not None:
        for row in rows:
            size = row.get("cohort_size")
            if size is None or int(size) < compiled.minimum_cohort_size:
                failures.append("cohort_below_minimum")
                break
        if any("obligor_id" in row or "facility_id" in row for row in rows):
            failures.append("cohort_leaked_identifier")

    # Re-assert the point-in-time bound on the data that came back, not just the predicate
    # that asked for it.
    for column in compiled.point_in_time_columns:
        for row in rows:
            observed = _as_date(row.get(column))
            if observed is not None and observed > plan.as_of_date:
                failures.append(f"point_in_time_breach:{column}")
                break

    for row in rows:
        observed = _as_date(row.get("observation_date"))
        if observed is not None and not (plan.date_from <= observed <= plan.date_to):
            failures.append("observation_date_out_of_range")
            break

    if failures:
        raise ResultValidationError(", ".join(sorted(set(failures))))

    missing = {
        column: sum(1 for row in rows if row.get(column) is None)
        for column in compiled.selected_columns
    }
    return {
        "grain": compiled.grain,
        "grain_keys": grain_keys,
        "row_limit": maximum_rows,
        "point_in_time_columns": compiled.point_in_time_columns,
        "as_of_date": plan.as_of_date.isoformat(),
        "missing_values": {column: count for column, count in missing.items() if count},
        "passed": True,
    }


def _execute_with_limits(
    compiled: CompiledQuery, controls: dict[str, Any], database_path: Path | None = None
) -> list[dict[str, Any]]:
    """Run the approved query under memory, time and row limits.

    DuckDB has no statement_timeout setting, so the wall-clock bound is enforced with a
    timer that calls interrupt() on the connection.
    """
    timeout = float(controls.get("statement_timeout_seconds", settings.query_timeout_seconds))
    connection = duckdb.connect(str(database_path or settings.database_path), read_only=True)
    watchdog = threading.Timer(timeout, connection.interrupt)
    try:
        # Bound the query's footprint so a wide scan cannot pull the dataset into memory.
        connection.execute("SET memory_limit=?", [str(controls.get("memory_limit", "512MB"))])
        connection.execute("SET threads=?", [int(controls.get("threads", 2))])
        connection.execute("SET enable_external_access=false")

        watchdog.start()
        cursor = connection.execute(compiled.sql, compiled.parameters)
        columns = [item[0] for item in cursor.description]
        return [
            {column: _json_safe(value) for column, value in zip(columns, row, strict=True)}
            for row in cursor.fetchall()
        ]
    finally:
        watchdog.cancel()
        connection.close()


@app.get("/internal/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "role": settings.service_role,
        "schema_registry_version": registry.version,
        "architecture_policy_version": policy.version,
    }


@app.post("/internal/v1/query/execute", dependencies=[Depends(authorize_service)])
def execute_query(request: InternalRequest) -> dict[str, Any]:
    plan = request.query_plan
    if not request.revision_id:
        raise HTTPException(409, "Approved revision required")
    try:
        policy.require(settings.service_role, "compile_parameterised_sql")
        policy.require(settings.service_role, "execute_read_only_sql")
        policy.require(settings.service_role, "return_validated_rows")
        compiled = compiler.compile(plan)
    except (ArchitecturePolicyError, QueryGuardError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if not settings.database_path.is_file():
        raise HTTPException(status_code=503, detail="Analytical database is not mounted")

    controls = registry.data["query_controls"]
    snapshot = snapshot_hash()
    try:
        store().consume(request.revision_id, request.reviewer_id, binding(plan, compiled, snapshot))
    except ReviewConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    try:
        rows = _execute_with_limits(compiled, controls)
    except duckdb.InterruptException as exc:
        raise HTTPException(
            status_code=504, detail="Approved query exceeded its time limit"
        ) from exc
    except duckdb.OutOfMemoryException as exc:
        raise HTTPException(
            status_code=507, detail="Approved query exceeded its memory limit"
        ) from exc
    except duckdb.Error as exc:
        # Never surface the raw database error: it carries paths and schema internals.
        raise HTTPException(status_code=422, detail="Approved query could not be executed") from exc

    if snapshot_hash() != snapshot:
        raise HTTPException(409, "Snapshot changed during execution")
    try:
        validation = validate_result(rows, compiled, plan, controls)
    except ResultValidationError as exc:
        # Categories only. The values that failed are not echoed back.
        raise HTTPException(
            status_code=422,
            detail=f"Result failed integrity validation: {exc}",
        ) from exc

    return {
        "schema_registry_version": registry.version,
        "source_table": compiled.source_table,
        "grain": compiled.grain,
        "selected_columns": compiled.selected_columns,
        "row_count": len(rows),
        "validation": validation,
        "rows": rows,
    }
