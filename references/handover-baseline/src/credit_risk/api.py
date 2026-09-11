from __future__ import annotations

import hashlib
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from credit_risk.guardrails import validate_input
from credit_risk.architecture_policy import ArchitecturePolicy, ArchitecturePolicyError
from credit_risk.query_guard import GuardedQueryCompiler, QueryGuardError, SchemaRegistry
from credit_risk.schemas import QueryPlan
from credit_risk.settings import settings


app = FastAPI(title="Credit Risk Fine-Tuning Development API", version="0.1.0")
policy = ArchitecturePolicy(settings.architecture_policy, settings.architecture_policy_version)
registry = SchemaRegistry(settings.schema_registry, settings.schema_registry_version)
compiler = GuardedQueryCompiler(registry)


class QueryValidationRequest(BaseModel):
    user_text: str
    query_plan: QueryPlan


def build_sql_review_packet(compiled) -> dict:
    """Expose an auditable SQL template without exposing sensitive parameter values."""
    query_hash = hashlib.sha256(
        f"{registry.version}:{compiled.sql}".encode("utf-8")
    ).hexdigest()
    return {
        "query_hash": query_hash,
        "parameterised_sql": compiled.sql,
        "parameter_placeholders": len(compiled.parameters),
        "parameter_values": ["***MASKED***" for _ in compiled.parameters],
        "source_table": compiled.source_table,
        "selected_columns": compiled.selected_columns,
        "schema_registry_version": registry.version,
        "editable": False,
        "execution_status": "NOT_EXECUTED",
        "review_instruction": (
            "Verify grain, tables, columns, filters, joins and calculations. "
            "Submit corrections as structured feedback; edited SQL is never executed directly."
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "role": settings.service_role,
        "schema_registry_version": registry.version,
        "architecture_policy_version": policy.version,
    }


@app.post("/v1/query/validate")
def validate_query(request: QueryValidationRequest) -> dict:
    input_result = validate_input(request.user_text, request.query_plan)
    if not input_result.passed:
        raise HTTPException(status_code=422, detail=input_result.failures)
    try:
        policy.require(settings.service_role, "validate_query_plan")
        policy.require(settings.service_role, "view_parameterised_sql")
        compiled = compiler.compile(request.query_plan)
    except (ArchitecturePolicyError, QueryGuardError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "status": "approved",
        "review_required": True,
        "sql_review": build_sql_review_packet(compiled),
    }


@app.post("/v1/query/fetch")
def fetch_data(request: QueryValidationRequest) -> dict:
    input_result = validate_input(request.user_text, request.query_plan)
    if not input_result.passed:
        raise HTTPException(status_code=422, detail=input_result.failures)
    try:
        policy.require(settings.service_role, "request_data_service")
        compiler.compile(request.query_plan)
    except (ArchitecturePolicyError, QueryGuardError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        with httpx.Client(timeout=settings.query_timeout_seconds) as client:
            response = client.post(
                f"{settings.data_service_url}/internal/v1/query/execute",
                headers={"x-service-token": settings.service_token},
                json=request.query_plan.model_dump(mode="json"),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Restricted data service unavailable") from exc



def run() -> None:
    import uvicorn

    uvicorn.run("credit_risk.api:app", host="127.0.0.1", port=8080, reload=False)

