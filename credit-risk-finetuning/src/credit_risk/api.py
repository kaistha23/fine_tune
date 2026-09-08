from __future__ import annotations

import hashlib
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from credit_risk.architecture_policy import ArchitecturePolicy, ArchitecturePolicyError
from credit_risk.factsheet import FactsheetError, build_factsheet
from credit_risk.feedback_store import FeedbackStore
from credit_risk.guardrails import validate_input, validate_output
from credit_risk.omlx_client import OMLXClient
from credit_risk.query_guard import (
    CompiledQuery,
    GuardedQueryCompiler,
    QueryGuardError,
    SchemaRegistry,
)
from credit_risk.rag.filters import AccessPolicyError, RetrievalPolicy
from credit_risk.rag.retriever import PolicyRetriever
from credit_risk.rag.schemas import AccessContext
from credit_risk.risk_tiers import gate
from credit_risk.schemas import EntityLevel, FeedbackRecord, QueryPlan
from credit_risk.settings import settings

app = FastAPI(title="Credit Risk Fine-Tuning Development API", version="0.1.0")
policy = ArchitecturePolicy(settings.architecture_policy, settings.architecture_policy_version)
registry = SchemaRegistry(settings.schema_registry, settings.schema_registry_version)
compiler = GuardedQueryCompiler(registry)
feedback_store = FeedbackStore(settings.feedback_path)
retrieval_policy = RetrievalPolicy(settings.retrieval_policy)
retriever = PolicyRetriever(retrieval_policy)
# Replaced in tests and wherever a real index or a served adapter is available.
model_client = OMLXClient(settings.omlx_base_url, settings.omlx_model)

# SQL review defects are schema, query or calculation problems. Routing them to the
# adapter would retrain the model for a bug in the query layer, which is exactly what the
# feedback plan's "fix the right component" rule forbids.
SQL_REVIEW_ROOT_CAUSES = ("schema", "query", "calculation", "data")


class QueryValidationRequest(BaseModel):
    user_text: str
    query_plan: QueryPlan


class AnalysisRequest(BaseModel):
    """A question about one obligor, answered from validated data and cited evidence."""

    user_text: str
    query_plan: QueryPlan
    reviewer_role: str = "credit_analyst"


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
    def require_comment_on_rejection(self) -> SqlReviewRequest:
        if self.decision == "rejected" and not self.comment:
            raise ValueError("comment is required when a SQL review is rejected")
        return self


def compute_query_hash(compiled: CompiledQuery) -> str:
    return hashlib.sha256(
        f"{registry.version}:{compiled.sql}".encode()
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
        # Cohort queries: the reviewer has to see what was grouped, what was aggregated and
        # what the suppression floor was, or "grain" alone tells them nothing about whether
        # the aggregate is safe to release.
        "group_by": compiled.group_by,
        "aggregations": compiled.aggregations,
        "minimum_cohort_size": compiled.minimum_cohort_size,
        "schema_registry_version": registry.version,
        "architecture_policy_version": policy.version,
        "editable": False,
        "execution_status": "NOT_EXECUTED",
        "review_instruction": (
            "Verify grain, tables, columns, filters, joins, governed calculations and, "
            "for a cohort query, the group_by dimensions and cohort-size floor. "
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


def _request_rows(request: QueryValidationRequest) -> dict:
    """Validate the plan, then ask the restricted data service for approved rows.

    The API never touches the database. It sends a validated plan and receives rows the
    data service has already checked against the grain and filters it approved.
    """
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


@app.post("/v1/query/fetch")
def fetch_data(request: QueryValidationRequest) -> dict:
    return _request_rows(request)


def _reject_cohort_plan(plan: QueryPlan) -> None:
    """Refuse a cohort plan before it reaches the data service.

    build_factsheet refuses it too, but only after the rows have been fetched. Checking
    here means a request that cannot succeed does not first pull obligor data across the
    service boundary.
    """
    if plan.entity_level == EntityLevel.PORTFOLIO:
        raise HTTPException(
            status_code=422,
            detail=("A factsheet is an obligor-level artefact and cannot be built from a "
                    "portfolio cohort; use /v1/query/fetch for cohort results"),
        )


@app.post("/v1/factsheet")
def factsheet(request: QueryValidationRequest) -> dict:
    """Approved rows in, compact factsheet out.

    This is the boundary the plans draw: deterministic Python owns every number, and the
    model is only ever shown this factsheet - never the raw monthly rows.
    """
    _reject_cohort_plan(request.query_plan)
    payload = _request_rows(request)
    try:
        sheet = build_factsheet(payload["rows"], request.query_plan)
    except FactsheetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "schema_registry_version": payload["schema_registry_version"],
        "source_table": payload["source_table"],
        "row_count": payload["row_count"],
        "validation": payload["validation"],
        "factsheet": sheet.model_dump(mode="json"),
    }


def run() -> None:
    import uvicorn

    uvicorn.run("credit_risk.api:app", host="127.0.0.1", port=8080, reload=False)


@app.post("/v1/analyse")
def analyse(request: AnalysisRequest) -> dict:
    """The full guarded path: rows -> factsheet -> evidence -> model -> gate.

    Each stage can refuse. The model is only reached once the data is validated and the
    evidence has passed its own filters, and its answer is checked against that evidence
    before anyone sees it.
    """
    _reject_cohort_plan(request.query_plan)
    fetch_request = QueryValidationRequest(
        user_text=request.user_text, query_plan=request.query_plan)
    payload = _request_rows(fetch_request)

    try:
        sheet = build_factsheet(payload["rows"], request.query_plan)
    except FactsheetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        context = AccessContext(
            jurisdiction=request.query_plan.jurisdiction,
            role=request.reviewer_role,
            as_of_date=request.query_plan.as_of_date,
            portfolio=request.query_plan.portfolio.value,
        )
        retrieval = retriever.retrieve(request.user_text, context)
    except AccessPolicyError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    evidence = retrieval["evidence"]
    if retrieval["answer_status"] == "INSUFFICIENT_EVIDENCE":
        # Abstention is a correct outcome, not an error. The factsheet is still returned
        # so the analyst can see what the numbers say without a cited narrative.
        return {
            "answer_status": "INSUFFICIENT_EVIDENCE",
            "factsheet": sheet.model_dump(mode="json"),
            "retrieval": {k: v for k, v in retrieval.items() if k != "evidence"},
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "response": None,
            "action_control": {"release": "blocked", "risk_tier": "medium",
                               "human_approval_required": True, "released": False,
                               "reasons": ["insufficient_evidence"]},
        }

    try:
        policy.require(settings.service_role, "call_model")
        response = model_client.generate_credit_response(
            request.user_text, sheet.model_dump(mode="json"), evidence)
    except ArchitecturePolicyError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        # Never surface the model host's address or its raw error.
        raise HTTPException(status_code=502, detail="Model service unavailable") from exc

    try:
        policy.require(settings.service_role, "validate_model_output")
    except ArchitecturePolicyError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    output_check = validate_output(response, evidence)
    control = gate(response, request.query_plan.analysis_type)
    if not output_check.passed:
        # An unsupported or miscited claim is never released, whatever the tier.
        control = {**control, "released": False, "release": "blocked",
                   "human_approval_required": True,
                   "reasons": sorted(set(control["reasons"] + output_check.failures))}

    return {
        "answer_status": response.answer_status.value,
        "factsheet": sheet.model_dump(mode="json"),
        "retrieval": {k: v for k, v in retrieval.items() if k != "evidence"},
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "response": response.model_dump(mode="json"),
        "output_guardrail": {"passed": output_check.passed,
                             "failures": output_check.failures},
        "action_control": control,
    }
