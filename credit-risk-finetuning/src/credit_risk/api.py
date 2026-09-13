from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from credit_risk.architecture_policy import ArchitecturePolicy, ArchitecturePolicyError
from credit_risk.auth import authenticate, current_reviewer, principal
from credit_risk.factsheet import FactsheetError, build_factsheet
from credit_risk.feedback import REMEDIATION_ROUTES
from credit_risk.feedback_store import FeedbackStore
from credit_risk.guardrails import validate_input, validate_output
from credit_risk.interaction_store import InteractionNotFound, InteractionStore
from credit_risk.omlx_client import OMLXClient
from credit_risk.prompts import PROMPT_VERSION
from credit_risk.query_guard import (
    QueryGuardError,
    SchemaRegistry,
)
from credit_risk.rag.factory import build_retriever
from credit_risk.rag.factory import describe as describe_retrieval
from credit_risk.rag.filters import AccessPolicyError, RetrievalPolicy
from credit_risk.rag.schemas import AccessContext
from credit_risk.risk_tiers import gate
from credit_risk.schemas import (
    AnswerStatus,
    CreditResponse,
    EntityLevel,
    Evidence,
    FeedbackRecord,
    InteractionRecord,
    QueryPlan,
)
from credit_risk.settings import settings

app = FastAPI(title="Credit Risk Fine-Tuning Development API", version="0.1.0")
policy = ArchitecturePolicy(settings.architecture_policy, settings.architecture_policy_version)
registry = SchemaRegistry(settings.schema_registry, settings.schema_registry_version)
feedback_store = FeedbackStore(settings.audit_store_path)
interaction_store = InteractionStore(settings.audit_store_path)
retrieval_policy = RetrievalPolicy(settings.retrieval_policy)
# Built from settings rather than defaulted: PolicyRetriever's default index is an empty
# in-memory one, so the deployed API used to retrieve nothing from the Qdrant service it
# was wired to and never opened.
retriever = build_retriever(settings, retrieval_policy)
# Replaced in tests and wherever a real index or a served adapter is available.
model_client = OMLXClient(
    settings.omlx_base_url, settings.omlx_model, api_key=settings.omlx_api_key
)

# SQL review defects are schema, query or calculation problems. Routing them to the
# adapter would retrain the model for a bug in the query layer, which is exactly what the
# feedback plan's "fix the right component" rule forbids.
SQL_REVIEW_ROOT_CAUSES = ("schema", "query", "calculation", "data")


@app.middleware("http")
async def reviewer_auth(request: Request, call_next):
    if request.url.path.startswith("/v1/"):
        try:
            who = authenticate(request.headers.get("authorization", ""))
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        token = principal.set(who)
        try:
            return await call_next(request)
        finally:
            principal.reset(token)
    return await call_next(request)


@app.get("/review", response_class=HTMLResponse)
def review_page():
    return (Path(__file__).parent / "static" / "review.html").read_text()


class QueryValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_text: str = Field(max_length=16000)
    query_plan: QueryPlan
    revision_id: str | None = None


class AnalysisRequest(QueryValidationRequest):
    pass


class SqlReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision_id: str
    decision: Literal["approved", "rejected"]
    comment: str = Field(default="", max_length=4000)
    corrected_query_plan: dict | None = None


def validate_plan(plan):
    policy.require(settings.service_role, "validate_query_plan")
    controls = registry.data["query_controls"]
    table_name = (
        "facility_monthly" if plan.entity_level == EntityLevel.FACILITY else "obligor_monthly"
    )
    if table_name not in controls["allowed_tables"]:
        raise QueryGuardError("Table not approved")
    table = registry.data["tables"][table_name]
    if plan.portfolio.value not in table["portfolios"]:
        raise QueryGuardError("Portfolio not approved")
    for metric in plan.metrics:
        registry.validate_metric(metric, plan.portfolio.value, table["grain"])


def data_call(operation, payload):
    policy.require(settings.service_role, "request_data_service")
    try:
        with httpx.Client(timeout=settings.query_timeout_seconds) as client:
            response = client.post(
                f"{settings.data_service_url}/internal/v1/query/{operation}",
                headers={"x-service-token": settings.service_token},
                json={**payload, "reviewer_id": current_reviewer()["id"]},
            )
        if response.status_code >= 400:
            raise HTTPException(
                response.status_code, response.json().get("detail", "Data request refused")
            )
        return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Restricted data service unavailable") from exc


@app.get("/health")
def health() -> dict[str, str]:
    readiness = "offline_test" if settings.offline_test_mode else "ready"
    if not settings.offline_test_mode:
        try:
            retriever.index.embedder.embed_query("retrieval readiness probe")
            if settings.qdrant_url:
                for collection in retrieval_policy.data["namespaces"].values():
                    if not retriever.index.client.collection_exists(collection):
                        raise ValueError("Unindexed collection")
                    retriever.index._check_signature(collection)
        except Exception as exc:
            raise HTTPException(503, "Retrieval backend is not ready") from exc
    return {
        "retrieval_readiness": readiness,
        "status": "ok",
        "role": settings.service_role,
        "schema_registry_version": registry.version,
        "architecture_policy_version": policy.version,
        # An operator must be able to tell a real backend from the offline fallback
        # without reading the container environment: the fallback answers every question
        # with no evidence and is indistinguishable from a quiet corpus.
        **{f"retrieval_{k}": v for k, v in describe_retrieval(settings).items()},
    }


@app.post("/v1/query/validate")
def validate_query(request: QueryValidationRequest):
    check = validate_input(request.user_text, request.query_plan)
    if not check.passed:
        raise HTTPException(422, check.failures)
    try:
        validate_plan(request.query_plan)
    except (QueryGuardError, ArchitecturePolicyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    packet = data_call("prepare", {"query_plan": request.query_plan.model_dump(mode="json")})
    return {"status": "pending", "review_required": True, "sql_review": packet}


@app.post("/v1/query/review")
def review_query(request: SqlReviewRequest):
    policy.require(settings.service_role, "record_sql_review")
    return data_call("review", request.model_dump(mode="json"))


def _request_rows(request: QueryValidationRequest):
    if not request.revision_id:
        raise HTTPException(409, "Approved revision required")
    check = validate_input(request.user_text, request.query_plan)
    if not check.passed:
        raise HTTPException(422, check.failures)
    try:
        validate_plan(request.query_plan)
    except (QueryGuardError, ArchitecturePolicyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return data_call(
        "execute",
        {
            "query_plan": request.query_plan.model_dump(mode="json"),
            "revision_id": request.revision_id,
        },
    )


class AnalysisFeedbackRequest(BaseModel):
    semantic_review: dict = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")
    """A reviewer's verdict on an answer the system actually served.

    eligible_for_training is absent by design. It is derived from the root cause and the
    presence of a revalidated correction, never supplied: a caller who could set it could
    put an unchecked target into the training set, which is the one thing the whole
    feedback pipeline exists to prevent.
    """

    quality_score: int = Field(ge=1, le=5)
    review_status: Literal["approved", "rejected"]
    interaction_id: str = Field(min_length=1, max_length=128)
    error_labels: list[
        Literal[
            "correct",
            "correct_style_change",
            "wrong_retrieval",
            "unsupported_claim",
            "numeric_error",
            "wrong_credit_interpretation",
            "wrong_jurisdiction",
            "missing_information_not_identified",
            "guardrail_failure",
            "should_have_abstained",
        ]
    ] = Field(min_length=1)
    root_cause: Literal[
        "data", "schema", "query", "calculation", "retrieval", "guardrail", "model_behaviour"
    ]
    # The corrected answer, as CreditResponse JSON. Required for a model-behaviour defect:
    # that is the only root cause retraining can fix, and it cannot be fixed without a
    # target to learn.
    corrected_output: str | None = None
    comment: str | None = Field(default=None, max_length=4000)


@app.post("/v1/analyse/feedback")
def analysis_feedback(request: AnalysisFeedbackRequest) -> dict:
    """Turn a reviewer's correction into a training-eligible record, or refuse it.

    The correction is revalidated against the evidence that was in context when the
    original answer was produced - the same rule the SQL review applies, for the same
    reason. An unchecked correction is worse than no correction: training on a target that
    cites evidence the model was never shown teaches it to cite from memory, which is
    precisely the failure the citation guardrail exists to catch.
    """
    policy.require(settings.service_role, "record_analysis_feedback")

    try:
        interaction = interaction_store.get(request.interaction_id)
    except InteractionNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail="Unknown interaction_id; it was never served or has been rotated out",
        ) from exc

    if interaction.reviewer_id and interaction.reviewer_id != current_reviewer()["id"]:
        raise HTTPException(403, "Interaction belongs to another reviewer")
    trainable_cause = request.root_cause == "model_behaviour"
    corrected_output = request.corrected_output
    revalidation: dict[str, object] = {"performed": False}

    if trainable_cause and not corrected_output:
        raise HTTPException(
            status_code=422,
            detail=(
                "A model_behaviour defect needs corrected_output: it is the only root "
                "cause retraining can fix, and it cannot be fixed without a target"
            ),
        )

    semantic_review = {}
    if request.semantic_review:
        semantic_review = {**request.semantic_review, "reviewer_id": current_reviewer()["id"],
                           "status": request.review_status}
    if corrected_output:
        try:
            corrected = CreditResponse.model_validate_json(corrected_output)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail="corrected_output is not a valid CreditResponse",
            ) from exc

        evidence = [Evidence.model_validate(item) for item in interaction.evidence]
        from credit_risk.guardrails import is_admissible_training_target

        check = is_admissible_training_target(
            corrected, evidence, interaction.factsheet, semantic_review
        )
        revalidation = {"performed": True, "passed": check.passed, "failures": check.failures}
        if not check.passed:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Correction failed revalidation and was not recorded as trainable",
                    "failures": check.failures,
                },
            )

    record = FeedbackRecord(
        semantic_review=semantic_review,
        interaction_id=request.interaction_id,
        reviewer_id=current_reviewer()["id"],
        review_status=request.review_status,
        quality_score=request.quality_score,
        model_id=interaction.model_id,
        adapter_version=interaction.adapter_version,
        dataset_version=interaction.dataset_version,
        portfolio=interaction.portfolio,
        task_type=interaction.task_type,
        input_case_id=interaction.case_id,
        retrieved_evidence_ids=[str(item.get("evidence_id", "")) for item in interaction.evidence],
        input_question=interaction.question,
        input_factsheet=interaction.factsheet,
        input_evidence=interaction.evidence,
        original_output=interaction.original_output or "",
        error_labels=list(request.error_labels),
        corrected_output=corrected_output,
        root_cause=request.root_cause,
        # Derived, never taken from the request: a model-behaviour defect with a
        # correction that passed revalidation, and nothing else.
        eligible_for_training=bool(
            trainable_cause
            and corrected_output
            and request.review_status == "approved"
            and request.quality_score >= 4
        ),
        sql_review_comment=request.comment,
    )
    feedback_store.append(record)

    return {
        "interaction_id": record.interaction_id,
        "root_cause": record.root_cause,
        "eligible_for_training": record.eligible_for_training,
        "revalidation": revalidation,
        # Says where a non-trainable defect actually goes, so a reviewer is not left
        # thinking a routed bug was ignored.
        "remediation_route": REMEDIATION_ROUTES.get(record.root_cause, "unrouted"),
        "recorded": True,
    }


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
            detail=(
                "A factsheet is an obligor-level artefact and cannot be built from a "
                "portfolio cohort; use /v1/query/fetch for cohort results"
            ),
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


def _record_interaction(
    request: AnalysisRequest,
    sheet,
    evidence: list,
    answer_status: str,
    original_output: str | None,
    guardrail_failures: list[str],
    release: str,
) -> str:
    """Persist what was served, and return the handle a correction will name.

    Recorded for abstentions too. An unnecessary abstention is a model-behaviour defect
    like any other, and if it cannot be corrected the loop only ever learns from answers
    the model was willing to give.

    The factsheet is stored, never the monthly rows it was built from: the factsheet is
    what the model actually saw, and the rows stay inside the data service.
    """
    policy.require(settings.service_role, "record_interaction")
    interaction_id = f"IX-{uuid4().hex[:16]}"
    interaction_store.append(
        InteractionRecord(
            interaction_id=interaction_id,
            reviewer_id=current_reviewer()["id"],
            model_id=settings.omlx_model,
            adapter_version=settings.adapter_version,
            dataset_version=settings.dataset_version,
            prompt_version=PROMPT_VERSION,
            portfolio=request.query_plan.portfolio,
            jurisdiction=request.query_plan.jurisdiction,
            task_type=request.query_plan.analysis_type,
            case_id=sheet.case_id,
            question=request.user_text,
            factsheet=sheet.model_dump(mode="json"),
            evidence=[item.model_dump(mode="json") for item in evidence],
            answer_status=AnswerStatus(answer_status),
            original_output=original_output,
            output_guardrail_failures=guardrail_failures,
            action_release=release,
        )
    )
    return interaction_id


@app.post("/v1/analyse")
def analyse(request: AnalysisRequest) -> dict:
    """The full guarded path: rows -> factsheet -> evidence -> model -> gate.

    Each stage can refuse. The model is only reached once the data is validated and the
    evidence has passed its own filters, and its answer is checked against that evidence
    before anyone sees it.
    """
    _reject_cohort_plan(request.query_plan)
    fetch_request = QueryValidationRequest(
        user_text=request.user_text, query_plan=request.query_plan, revision_id=request.revision_id
    )
    payload = _request_rows(fetch_request)

    try:
        sheet = build_factsheet(payload["rows"], request.query_plan)
    except FactsheetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        context = AccessContext(
            jurisdiction=request.query_plan.jurisdiction,
            role=current_reviewer()["role"],
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
            "interaction_id": _record_interaction(
                request,
                sheet,
                evidence,
                "INSUFFICIENT_EVIDENCE",
                None,
                ["insufficient_evidence"],
                "blocked",
            ),
            "answer_status": "INSUFFICIENT_EVIDENCE",
            "factsheet": sheet.model_dump(mode="json"),
            "retrieval": {k: v for k, v in retrieval.items() if k != "evidence"},
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "response": None,
            "action_control": {
                "release": "blocked",
                "risk_tier": "medium",
                "human_approval_required": True,
                "released": False,
                "reasons": ["insufficient_evidence"],
            },
        }

    try:
        policy.require(settings.service_role, "call_model")
        response = model_client.generate_credit_response(
            request.user_text, sheet.model_dump(mode="json"), evidence
        )
    except ArchitecturePolicyError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        # Never surface the model host's address or its raw error.
        raise HTTPException(status_code=502, detail="Model service unavailable") from exc

    try:
        policy.require(settings.service_role, "validate_model_output")
    except ArchitecturePolicyError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    output_check = validate_output(
        response, evidence, factsheet_case_id=sheet.case_id, factsheet=sheet.model_dump(mode="json")
    )
    control = gate(response, request.query_plan.analysis_type)
    if not output_check.passed:
        # An unsupported or miscited claim is never released, whatever the tier.
        control = {
            **control,
            "released": False,
            "release": "blocked",
            "human_approval_required": True,
            "reasons": sorted(set(control["reasons"] + output_check.failures)),
        }

    return {
        # The handle a correction names. Without it an analyst who sees a wrong answer has
        # nothing to attach the correction to, and the training loop has no input.
        "interaction_id": _record_interaction(
            request,
            sheet,
            evidence,
            response.answer_status.value,
            response.model_dump_json(),
            output_check.failures,
            control["release"],
        ),
        "answer_status": response.answer_status.value,
        "factsheet": sheet.model_dump(mode="json"),
        "retrieval": {k: v for k, v in retrieval.items() if k != "evidence"},
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "response": response.model_dump(mode="json"),
        "output_guardrail": {"passed": output_check.passed, "failures": output_check.failures},
        "action_control": control,
    }
