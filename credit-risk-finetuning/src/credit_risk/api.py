from __future__ import annotations

import hashlib
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from credit_risk.architecture_policy import ArchitecturePolicy, ArchitecturePolicyError
from credit_risk.feedback_store import FeedbackStore
from credit_risk.guardrails import validate_input
from credit_risk.query_guard import CompiledQuery, GuardedQueryCompiler, QueryGuardError, SchemaRegistry
from credit_risk.schemas import FeedbackRecord, QueryPlan
from credit_risk.settings import settings


app = FastAPI(title="Credit Risk Fine-Tuning Development API", version="0.1.0")
policy = ArchitecturePolicy(settings.architecture_policy, settings.architecture_policy_version)
registry = SchemaRegistry(settings.schema_registry, settings.schema_registry_version)
compiler = GuardedQueryCompiler(registry)
feedback_store = FeedbackStore(settings.feedback_path)

# SQL review defects are schema, query or calculation problems. Routing them to the
# adapter would retrain the model for a bug in the query layer, which is exactly what the
# feedback plan's "fix the right component" rule forbids.
SQL_REVIEW_ROOT_CAUSES = ("schema", "query", "calculation", "data")


class QueryValidationRequest(BaseModel):
    user_text: str
    query_plan: QueryPlan


class SqlReviewRequest(BaseModel):
    """A reviewer's decision on a parameterised SQL template.

    The reviewer never submits SQL. They submit a decision and, when rejecting, a
    corrected structured query plan which is revalidated from scratch.
    """
    interaction_id: str = Field(min_length=1, max_length=128)
    query_hash: str = Field(min_length=64, max_length=64)
    reviewed_query_plan: QueryPlan
    decision: Literal["approved", "rejected"]
    comment: str | None = None
    corrected_query_plan: QueryPlan | None = None
    error_labels: list[Literal[
        "wrong_table", "wrong_column", "wrong_join", "wrong_grain", "wrong_filter",
        "wrong_calculation", "sql_review_rejected",
    ]] = Field(default_factory=list)
    root_cause: Literal["data", "schema", "query", "calculation"] = "query"
    model_id: str = "n/a"
    adapter_version: str = "n/a"
    dataset_version: str = "n/a"
    task_type: str = "sql_review"

    @model_validator(mode="after")
    def require_comment_on_rejection(self) -> "SqlReviewRequest":
        if self.decision == "rejected" and not self.comment:
            raise ValueError("comment is required when a SQL review is rejected")
        return self


def compute_query_hash(compiled: CompiledQuery) -> str:
    return hashlib.sha256(
        f"{registry.version}:{compiled.sql}".encode("utf-8")
    ).hexdigest()


def build_sql_review_packet(compiled: CompiledQuery) -> dict:
    """Expose an auditable SQL template without exposing sensitive parameter values."""
    return {
        "query_hash": compute_query_hash(compiled),
        "parameterised_sql": compiled.sql,
        "parameter_placeholders": len(compiled.parameters),
        "parameter_values": ["***MASKED***" for _ in compiled.parameters],
        "source_table": compiled.source_table,
        "grain": compiled.grain,
        "selected_columns": compiled.selected_columns,
        "joins": compiled.joins,
        "filters": compiled.filters,
        "governed_calculations": compiled.governed_calculations,
        "point_in_time_columns": compiled.point_in_time_columns,
        "schema_registry_version": registry.version,
        "architecture_policy_version": policy.version,
        "editable": False,
        "execution_status": "NOT_EXECUTED",
        "review_instruction": (
            "Verify grain, tables, columns, filters, joins and governed calculations. "
            "Submit corrections as a structured query plan to /v1/query/review; edited SQL "
            "is never executed directly."
        ),
    }


def _compile_for_review(plan: QueryPlan) -> CompiledQuery:
    policy.require(settings.service_role, "validate_query_plan")
    policy.require(settings.service_role, "compile_parameterised_sql_for_review")
    policy.require(settings.service_role, "view_parameterised_sql")
    return compiler.compile(plan)


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
        compiled = _compile_for_review(request.query_plan)
    except (ArchitecturePolicyError, QueryGuardError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "status": "approved",
        "review_required": True,
        "sql_review": build_sql_review_packet(compiled),
    }


@app.post("/v1/query/review")
def review_query(request: SqlReviewRequest) -> dict:
    """Record a reviewer's decision, and revalidate any correction from scratch.

    Manually edited SQL is never accepted or executed. A correction is a structured query
    plan, and it goes through the identical default-deny policy and registry cycle that
    the original plan did before it can be used.
    """
    try:
        policy.require(settings.service_role, "record_sql_review")
        reviewed = _compile_for_review(request.reviewed_query_plan)
    except (ArchitecturePolicyError, QueryGuardError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # The reviewer must be deciding on the template they were actually shown.
    reviewed_hash = compute_query_hash(reviewed)
    if reviewed_hash != request.query_hash:
        raise HTTPException(
            status_code=409,
            detail="query_hash does not match the submitted plan; re-validate before reviewing",
        )

    corrected_packet = None
    corrected_sql = None
    if request.corrected_query_plan is not None:
        try:
            corrected = _compile_for_review(request.corrected_query_plan)
        except (ArchitecturePolicyError, QueryGuardError) as exc:
            # A correction that fails validation is itself feedback, not an accepted plan.
            raise HTTPException(
                status_code=422,
                detail=f"Corrected query plan failed revalidation: {exc}",
            ) from exc
        corrected_packet = build_sql_review_packet(corrected)
        corrected_sql = corrected.sql

    labels = list(request.error_labels)
    if request.decision == "rejected" and not labels:
        labels = ["sql_review_rejected"]

    record = FeedbackRecord(
        interaction_id=request.interaction_id,
        model_id=request.model_id,
        adapter_version=request.adapter_version,
        dataset_version=request.dataset_version,
        portfolio=request.reviewed_query_plan.portfolio,
        task_type=request.task_type,
        input_case_id=request.reviewed_query_plan.obligor_id,
        original_output=reviewed.sql,
        error_labels=labels,
        corrected_output=corrected_sql,
        root_cause=request.root_cause,
        query_hash=reviewed_hash,
        sql_review_status=request.decision,
        sql_review_comment=request.comment,
        corrected_query_plan=request.corrected_query_plan,
        # Never true for a SQL review: these are query-layer defects, and training the
        # adapter on them would fix the wrong component.
        eligible_for_training=False,
    )
    feedback_store.append(record)

    return {
        "status": "recorded",
        "interaction_id": record.interaction_id,
        "sql_review_status": record.sql_review_status,
        "root_cause": record.root_cause,
        "routed_to": "schema_and_query_backlog",
        "eligible_for_training": record.eligible_for_training,
        "revalidated": corrected_packet is not None,
        "corrected_sql_review": corrected_packet,
    }


@app.post("/v1/query/fetch")
def fetch_data(request: QueryValidationRequest) -> dict:
    input_result = validate_input(request.user_text, request.query_plan)
    if not input_result.passed:
        raise HTTPException(status_code=422, detail=input_result.failures)
    try:
        policy.require(settings.service_role, "request_data_service")
        _compile_for_review(request.query_plan)
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
