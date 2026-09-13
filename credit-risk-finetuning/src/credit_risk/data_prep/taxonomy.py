"""Controlled task and situation taxonomy for credit-risk datasets."""

from enum import StrEnum


class TaskType(StrEnum):
    FACTSHEET_QA = "factsheet_qa"
    FACTSHEET = "factsheet"
    EMAIL_DRAFT = "email_draft"
    CLAIM_VERIFICATION = "claim_verification"
    QUERY_PLAN = "query_plan"
    METRIC_INTERPRETATION = "metric_interpretation"
    FIELD_INTERPRETATION = "field_interpretation"
    EWS_ANALYSIS = "ews_analysis"
    CREDIT_DETERIORATION = "credit_deterioration"
    POLICY_QA = "policy_qa"


class Situation(StrEnum):
    BASE = "base"
    NEAR_MISS = "near_miss"
    MITIGANT = "mitigant"
    GRAIN_TRAP = "grain_trap"
    SUPERSEDED_POLICY = "superseded_policy"
    WRONG_JURISDICTION = "wrong_jurisdiction"
    MISSING_FIELD = "missing_field"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INJECTION = "injection"


TASK_TYPE_ALIASES = {
    # Historical feedback used the shorter name. Canonicalize it while reading so
    # existing reviewed records remain usable without weakening the controlled enum.
    "ews": TaskType.EWS_ANALYSIS.value,
}


def normalize_task_type(value: str) -> str:
    return TaskType(TASK_TYPE_ALIASES.get(value, value)).value


def dimensions(record):
    case = record.get("case", record)
    task_type = record.get("task_type", case.get("task_type"))
    situation = record.get("situation", case.get("situation"))
    return {
        "task_type": normalize_task_type(task_type),
        "portfolio": case["portfolio"],
        "jurisdiction": case["jurisdiction"],
        "situation": Situation(situation).value,
    }
