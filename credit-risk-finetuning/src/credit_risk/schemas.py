from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def compact_json_schema(value: Any) -> Any:
    """Remove prompt-irrelevant annotations while preserving validation semantics."""

    if isinstance(value, dict):
        return {
            key: compact_json_schema(item)
            for key, item in value.items()
            if key not in {"title", "description", "default"}
        }
    if isinstance(value, list):
        return [compact_json_schema(item) for item in value]
    return value


class Portfolio(StrEnum):
    RETAIL = "retail"
    SME = "sme"
    CORPORATE = "corporate"


class Jurisdiction(StrEnum):
    SAMA = "SAMA"
    CBUAE = "CBUAE"


class EntityLevel(StrEnum):
    OBLIGOR = "obligor"
    FACILITY = "facility"
    # A cohort over a portfolio, never a named obligor. See QueryPlan.validate_dates_and_grain.
    PORTFOLIO = "portfolio"


class AnswerStatus(StrEnum):
    ANSWERED = "ANSWERED"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ESCALATE = "ESCALATE"


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    portfolio: Portfolio
    jurisdiction: Jurisdiction
    entity_level: EntityLevel = EntityLevel.OBLIGOR
    obligor_id: str | None = Field(default=None, max_length=128)
    facility_id: str | None = Field(default=None, max_length=128)
    # Cohort dimensions, allowlisted in the registry. Empty for obligor and facility plans.
    group_by: list[str] = Field(default_factory=list, max_length=4)
    cohort_aggregation: Literal["count", "sum", "avg"] = "avg"
    date_from: date
    date_to: date
    as_of_date: date
    metrics: list[str] = Field(min_length=1, max_length=30)
    analysis_type: Literal[
        "credit_deterioration", "ews_analysis", "policy_qa", "email_draft", "factsheet"
    ] = "credit_deterioration"

    @model_validator(mode="after")
    def validate_dates_and_grain(self) -> QueryPlan:
        if self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        # as_of_date is the point-in-time anchor: it is what the analyst is allowed to know.
        # Observing past it would leak data that did not exist at decision time.
        if self.date_to > self.as_of_date:
            raise ValueError("date_to must not be after as_of_date")
        if self.entity_level == EntityLevel.FACILITY and not self.facility_id:
            raise ValueError("facility_id is required for facility-level queries")

        if self.entity_level == EntityLevel.PORTFOLIO:
            # A cohort query that names an obligor is an individual extract with a GROUP BY
            # bolted on. Refusing it here means the cohort-size floor cannot be sidestepped.
            if self.obligor_id or self.facility_id:
                raise ValueError("portfolio-level queries must not name an obligor or facility")
        else:
            if not self.obligor_id:
                raise ValueError("obligor_id is required for obligor and facility queries")
            if self.group_by:
                raise ValueError("group_by applies to portfolio-level queries only")
        return self


class MetricValue(BaseModel):
    value: float | int | str | bool | None
    formula_id: str | None = None
    source_columns: list[str] = Field(default_factory=list)
    missing_data_flag: bool = False
    validation_status: Literal["valid", "warning", "invalid"] = "valid"


class CreditFactsheet(BaseModel):
    schema_version: str = "credit_factsheet_v1"
    case_id: str
    obligor_id: str
    portfolio: Portfolio
    jurisdiction: Jurisdiction
    as_of_date: date
    observation_months: int = Field(ge=1, le=60)
    current_position: dict[str, Any]
    calculated_metrics: dict[str, MetricValue]
    trends: dict[str, Any] = Field(default_factory=dict)
    events: list[str] = Field(default_factory=list)
    model_outputs: dict[str, Any] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)
    data_quality_flags: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    evidence_id: str
    jurisdiction: Jurisdiction
    document_id: str
    document_version: str
    section: str
    text: str
    score: float = Field(ge=0, le=1)
    effective_from: date | None = None
    effective_to: date | None = None


class SupportedClaim(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class InferenceClaim(BaseModel):
    statement: str
    basis: str
    confidence: float = Field(ge=0, le=1)


class CreditConclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conclusion_type: str = Field(min_length=1, max_length=100)
    value: bool | int | float | str
    severity: Literal["low", "medium", "high", "critical"]
    evidence_ids: list[str] = Field(default_factory=list)


class RiskDriverDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    driver: str = Field(min_length=1, max_length=100)
    observed_value: bool | int | float | str | None = None
    prior_value: bool | int | float | str | None = None
    unit: str | None = Field(default=None, max_length=32)
    currency: str | None = Field(default=None, max_length=3)
    direction: Literal["improving", "stable", "deteriorating", "unknown"]
    severity: Literal["low", "medium", "high", "critical"]
    as_of_date: date
    evidence_ids: list[str] = Field(default_factory=list)


class MissingInformationDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=500)


class RecommendationDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str = Field(min_length=1, max_length=200)
    rationale_evidence_ids: list[str] = Field(default_factory=list)


class CreditResponse(BaseModel):
    answer_status: AnswerStatus
    executive_summary: str
    facts: list[SupportedClaim] = Field(default_factory=list)
    inferences: list[InferenceClaim] = Field(default_factory=list)
    risk_drivers: list[str] = Field(default_factory=list)
    mitigants: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommendation: str = ""
    human_approval_required: bool = True
    conclusions: list[CreditConclusion] = Field(default_factory=list)
    risk_driver_details: list[RiskDriverDetail] = Field(default_factory=list)
    missing_information_details: list[MissingInformationDetail] = Field(default_factory=list)
    recommendation_detail: RecommendationDetail | None = None


class InteractionRecord(BaseModel):
    reviewer_id: str = ""
    """What the model was shown and what it answered, kept so a correction can be trained on.

    Without this the feedback loop has no input: FeedbackRecord carried only an
    input_case_id, and an identifier cannot reconstruct a training example. Storing the
    factsheet rather than the rows it came from is deliberate - the factsheet is the
    derived artefact the model actually saw, and the raw monthly rows never leave the data
    service.
    """

    interaction_id: str = Field(min_length=1, max_length=128)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    adapter_version: str
    dataset_version: str
    prompt_version: str
    portfolio: Portfolio
    jurisdiction: Jurisdiction
    task_type: str
    case_id: str
    question: str
    factsheet: dict[str, Any]
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    answer_status: AnswerStatus
    # The model's answer as served, serialised. None when the path abstained before the
    # model was reached, which is itself a correctable behaviour.
    original_output: str | None = None
    output_guardrail_failures: list[str] = Field(default_factory=list)
    action_release: str = ""


class FeedbackRecord(BaseModel):
    attempted_query_plan: dict[str, Any] | None = None
    reviewed_query_plan: dict[str, Any] | None = None
    sql_review_packet: dict[str, Any] | None = None
    reviewer_id: str = ""
    review_status: Literal["pending", "approved", "rejected"] = "pending"
    quality_score: int = Field(default=0, ge=0, le=5)
    data_classification: Literal["synthetic", "masked"] = "synthetic"
    interaction_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    adapter_version: str
    dataset_version: str
    portfolio: Portfolio
    task_type: str
    input_case_id: str
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    # What the model was actually shown. Without these a feedback record cannot be turned
    # back into a training example: the batch builder had only input_case_id and emitted
    # "Case reference: FB-123" as the user turn, which teaches the model to produce a full
    # credit assessment from an identifier containing none of the case data. Storing the
    # shown context is also what lets a reviewer see which input produced the bad output.
    input_question: str = ""
    input_factsheet: dict[str, Any] | None = None
    input_evidence: list[dict[str, Any]] = Field(default_factory=list)
    original_output: str
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
            "wrong_table",
            "wrong_column",
            "wrong_join",
            "wrong_grain",
            "wrong_filter",
            "wrong_calculation",
            "sql_review_rejected",
        ]
    ]
    corrected_output: str | None = None
    root_cause: Literal[
        "data", "schema", "query", "calculation", "retrieval", "guardrail", "model_behaviour"
    ]
    query_hash: str | None = None
    sql_review_status: Literal["not_reviewed", "approved", "rejected"] = "not_reviewed"
    sql_review_comment: str | None = None
    corrected_query_plan: QueryPlan | None = None
    eligible_for_training: bool = False
    feedback_batch: str | None = None

    @model_validator(mode="after")
    def require_correction_for_training(self) -> FeedbackRecord:
        if self.eligible_for_training and not self.corrected_output:
            raise ValueError("corrected_output is required when eligible_for_training is true")
        if self.sql_review_status == "rejected" and not self.sql_review_comment:
            raise ValueError("sql_review_comment is required when SQL review is rejected")
        return self
