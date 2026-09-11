from __future__ import annotations

from datetime import date, datetime
from typing import Any

import duckdb
from fastapi import Depends, FastAPI, Header, HTTPException

from credit_risk.architecture_policy import ArchitecturePolicy, ArchitecturePolicyError
from credit_risk.query_guard import GuardedQueryCompiler, QueryGuardError, SchemaRegistry
from credit_risk.schemas import QueryPlan
from credit_risk.settings import settings


app = FastAPI(title="Restricted Credit Data Service", version="0.1.0", docs_url=None)
policy = ArchitecturePolicy(settings.architecture_policy, settings.architecture_policy_version)
registry = SchemaRegistry(settings.schema_registry, settings.schema_registry_version)
compiler = GuardedQueryCompiler(registry)


def authorize_service(x_service_token: str = Header(default="")) -> None:
    if x_service_token != settings.service_token:
        raise HTTPException(status_code=401, detail="Invalid internal service credential")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


@app.get("/internal/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "role": settings.service_role,
        "schema_registry_version": registry.version,
        "architecture_policy_version": policy.version,
    }


@app.post("/internal/v1/query/execute", dependencies=[Depends(authorize_service)])
def execute_query(plan: QueryPlan) -> dict[str, Any]:
    try:
        policy.require(settings.service_role, "compile_parameterised_sql")
        policy.require(settings.service_role, "execute_read_only_sql")
        compiled = compiler.compile(plan)
    except (ArchitecturePolicyError, QueryGuardError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if not settings.database_path.is_file():
        raise HTTPException(status_code=503, detail="Analytical database is not mounted")

    try:
        connection = duckdb.connect(str(settings.database_path), read_only=True)
        try:
            cursor = connection.execute(compiled.sql, compiled.parameters)
            columns = [item[0] for item in cursor.description]
            rows = [
                {column: _json_safe(value) for column, value in zip(columns, row, strict=True)}
                for row in cursor.fetchall()
            ]
        finally:
            connection.close()
    except duckdb.Error as exc:
        raise HTTPException(status_code=422, detail="Approved query could not be executed") from exc

    return {
        "schema_registry_version": registry.version,
        "source_table": compiled.source_table,
        "selected_columns": compiled.selected_columns,
        "row_count": len(rows),
        "rows": rows,
    }


