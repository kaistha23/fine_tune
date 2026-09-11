from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


class AnswerStatus(StrEnum):
    ANSWERED = "ANSWERED"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ESCALATE = "ESCALATE"


class QueryPlan(BaseModel):
    portfolio: Portfolio
    jurisdiction: Jurisdiction
    entity_level: EntityLevel = EntityLevel.OBLIGOR
    obligor_id: str = Field(min_length=1, max_length=128)
    facility_id: str | None = Field(default=None, max_length=128)
    date_from: date
    date_to: date
    metrics: list[str] = Field(min_length=1, max_length=30)
    analysis_type: Literal[
        "credit_deterioration", "ews_analysis", "policy_qa", "email_draft", "factsheet"
    ] = "credit_deterioration"

    @model_validator(mode="after")
    def validate_dates_and_grain(self) -> "QueryPlan":
        if self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        if self.entity_level == EntityLevel.FACILITY and not self.facility_id:
            raise ValueError("facility_id is required for facility-level queries")
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


class FeedbackRecord(BaseModel):
    interaction_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    adapter_version: str
    dataset_version: str
    portfolio: Portfolio
    task_type: str
    input_case_id: str
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    original_output: str
    error_labels: list[Literal[
        "correct", "correct_style_change", "wrong_retrieval", "unsupported_claim",
        "numeric_error", "wrong_credit_interpretation", "wrong_jurisdiction",
        "missing_information_not_identified", "guardrail_failure", "should_have_abstained",
        "wrong_table", "wrong_column", "wrong_join", "wrong_grain", "wrong_filter",
        "wrong_calculation", "sql_review_rejected"
    ]]
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
    def require_correction_for_training(self) -> "FeedbackRecord":
        if self.eligible_for_training and not self.corrected_output:
            raise ValueError("corrected_output is required when eligible_for_training is true")
        if self.sql_review_status == "rejected" and not self.sql_review_comment:
            raise ValueError("sql_review_comment is required when SQL review is rejected")
        return self

