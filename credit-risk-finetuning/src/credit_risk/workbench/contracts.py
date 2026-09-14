"""Phase-2 contracts and immutable dataset inspection; never synthesizes training data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from credit_risk.data_prep.taxonomy import Situation, TaskType
from credit_risk.prompts import SYSTEM_PROMPT
from credit_risk.review_store import digest
from credit_risk.schemas import CreditResponse, QueryPlan, compact_json_schema

Task = Literal["credit_analysis", "query_plan"]
Split = Literal["train", "validation", "test", "oot", "development"]
SPLITS = ("train", "validation", "test", "oot")
QUERY_PROMPT = "Return only a structured query plan matching the supplied JSON schema and registry. Use the requested entity, dates, jurisdiction and metrics. Never emit SQL text."


class TypedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_id: str = Field(min_length=1, max_length=128)
    metric: str = Field(min_length=1, max_length=128)
    value: bool | int | float | str | None
    unit: str | None = Field(default=None, max_length=32)
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    effective_date: str
    source_id: str = Field(min_length=1, max_length=256)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    task: Task
    task_type: TaskType | None = None
    situation: Situation = Situation.BASE
    split: Split
    question: str = Field(min_length=1)
    portfolio: Literal["retail", "sme", "corporate"]
    jurisdiction: Literal["SAMA", "CBUAE"]
    as_of_date: str
    facts: dict[str, Any] = Field(default_factory=dict)
    fact_records: list[TypedFact] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    rule_evaluations: list[dict] = Field(default_factory=list)
    target: dict | None = None
    expected: dict = Field(default_factory=dict)
    consistency_paths: list[str] = Field(default_factory=list)
    equivalence_id: str | None = None
    distinct_from: list[str] = Field(default_factory=list)
    sql_lineage: dict | None = None
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def legacy_taxonomy_default(self):
        if self.task_type is None:
            self.task_type = TaskType.QUERY_PLAN if self.task == "query_plan" else TaskType.FACTSHEET
        if self.task == "query_plan" and self.task_type != TaskType.QUERY_PLAN:
            raise ValueError("query_plan cases require query_plan task_type")
        if self.task != "query_plan" and self.task_type == TaskType.QUERY_PLAN:
            raise ValueError("query_plan task_type requires query_plan task")
        return self

    def context_hash(self):
        return digest(
            {
                "facts": self.facts,
                "fact_records": [fact.model_dump(mode="json") for fact in self.fact_records],
                "evidence": self.evidence,
                "rule_evaluations": self.rule_evaluations,
                "task": self.task,
                "task_type": self.task_type,
                "situation": self.situation,
                "portfolio": self.portfolio,
                "jurisdiction": self.jurisdiction,
                "as_of_date": self.as_of_date,
            }
        )


def default_version(task):
    schema = compact_json_schema(
        CreditResponse.model_json_schema()
        if task == "credit_analysis"
        else QueryPlan.model_json_schema()
    )
    schema["additionalProperties"] = False
    return {
        "task": task,
        "prompt": SYSTEM_PROMPT if task == "credit_analysis" else QUERY_PROMPT,
        "schema": schema,
        "parent": None,
        "name": "Initial local version",
    }


def messages(case, version):
    return [
        {"role": "system", "content": version["prompt"]},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": case.question,
                    "context": {
                        "factsheet": case.facts,
                        "fact_records": [fact.model_dump(mode="json") for fact in case.fact_records],
                        "evidence": case.evidence,
                        "rule_evaluations": case.rule_evaluations,
                    },
                    "response_schema": version["schema"],
                },
                ensure_ascii=False,
            ),
        },
    ]


def inspect_dataset(manifest_path):
    path = Path(manifest_path).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    raw = json.loads(path.read_text())
    if raw.get("format") not in ("credit-workbench-v1", "credit-workbench-v2") or raw.get(
        "task"
    ) not in (
        "credit_analysis",
        "query_plan",
    ):
        raise ValueError("Expected credit-workbench-v1/v2 and a supported task")
    v2 = raw["format"] == "credit-workbench-v2"
    if v2:
        for key in ("dataset_id", "dataset_version", "created_at"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError("V2 manifest requires " + key)
        datetime.fromisoformat(raw["created_at"])
    if set(raw.get("splits", {})) != set(SPLITS):
        raise ValueError(
            "Declare train, validation, test and oot explicitly (empty files are allowed)"
        )
    registry_entry = raw.get("schema_registry")
    if raw["task"] == "query_plan" and not registry_entry:
        raise ValueError("Query-plan datasets require a versioned schema_registry file")
    if registry_entry:
        registry_file = (path.parent / registry_entry["file"]).resolve()
        if not registry_file.is_relative_to(path.parent):
            raise ValueError("Registry escapes dataset directory")
        if hashlib.sha256(registry_file.read_bytes()).hexdigest() != registry_entry["sha256"]:
            raise ValueError("Schema registry checksum mismatch")
    cases = []
    seen = set()
    groups = {}
    hashes = {}
    equivalents = {}
    from datetime import date

    cutoff = date.fromisoformat(raw["oot_from"])
    for split, entry in raw["splits"].items():
        file = (path.parent / entry["file"]).resolve()
        if not file.is_relative_to(path.parent):
            raise ValueError("Dataset file escapes manifest directory")
        content = file.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("Dataset checksum mismatch")
        hashes[split] = entry["sha256"]
        for line in content.decode().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            case = Case.model_validate(payload)
            if case.split != split or case.task != raw["task"]:
                raise ValueError("Case task/split mismatch")
            if case.provenance.get("classification") not in ("synthetic", "masked"):
                raise ValueError("Case provenance must declare synthetic or masked classification")
            if v2:
                for key in ("source_snapshot_hash", "template_family", "transformations"):
                    if key not in case.provenance:
                        raise ValueError("V2 case provenance requires " + key)
                source_hash = case.provenance["source_snapshot_hash"]
                if not isinstance(source_hash, str) or len(source_hash) != 64:
                    raise ValueError("V2 source_snapshot_hash must be a SHA-256 hex digest")
                try:
                    int(source_hash, 16)
                except ValueError as exc:
                    raise ValueError("V2 source_snapshot_hash must be hexadecimal") from exc
                if not isinstance(case.provenance["transformations"], list):
                    raise ValueError("V2 transformations must be a list")
                if case.task == "credit_analysis" and "fact_records" not in payload:
                    raise ValueError("V2 credit cases must declare fact_records")
                if split in ("validation", "test", "oot") and not case.expected:
                    raise ValueError("V2 held-out cases require independent expected checks")
                if split in ("train", "validation") and case.target is None:
                    raise ValueError("V2 train/validation cases require targets")
            if case.case_id in seen:
                raise ValueError("Duplicate case ID")
            seen.add(case.case_id)
            if groups.setdefault(case.group_id, split) != split:
                raise ValueError("Group leakage between splits")
            family = case.provenance.get("template_family")
            if family and groups.setdefault("template:" + family, split) != split:
                raise ValueError("Template family leakage between splits")
            if split == "oot" and date.fromisoformat(case.as_of_date) < cutoff:
                raise ValueError("OOT case precedes cutoff")
            if split != "oot" and date.fromisoformat(case.as_of_date) >= cutoff:
                raise ValueError("Post-cutoff case outside OOT")
            if case.equivalence_id:
                signature = (case.context_hash(), split, case.consistency_paths, case.expected)
                if equivalents.setdefault(case.equivalence_id, signature) != signature:
                    raise ValueError(
                        "Equivalent questions have different contexts, expectations or splits"
                    )
            cases.append(case)
    if v2 and any(c.provenance.get("classification") == "masked" for c in cases):
        scan = raw.get("privacy_scan", {})
        if scan.get("status") != "passed" or not scan.get("scanner_version"):
            raise ValueError("Masked V2 data requires a passed versioned privacy scan")
    fingerprints = {}
    for case in cases:
        key = digest({"question": case.question.strip().casefold(), "context": case.context_hash()})
        if key in fingerprints:
            raise ValueError("Duplicate question/context")
        fingerprints[key] = case.case_id
        if any(i not in seen or i == case.case_id for i in case.distinct_from):
            raise ValueError("Negative control must reference another registered case")
    taxonomy_counts = {}
    for case in cases:
        key = "|".join(
            (case.task_type.value, case.portfolio, case.jurisdiction, case.situation.value)
        )
        taxonomy_counts[key] = taxonomy_counts.get(key, 0) + 1
    diversity = None
    if v2:
        from credit_risk.data_prep.diversity import diversity_report

        diversity = diversity_report([case.model_dump(mode="json") for case in cases])
        if diversity["violations"]:
            raise ValueError("Dataset diversity failed: " + ", ".join(diversity["violations"]))
    return {
        "path": str(path),
        "manifest": raw,
        "hash": digest(raw),
        "counts": {s: sum(c.split == s for c in cases) for s in SPLITS},
        "missing_targets": sum(
            c.target is None for c in cases if c.split in ("train", "validation")
        ),
        "missing_expectations": sum(not c.expected for c in cases if c.split != "train"),
        "token_lengths": None,
        "taxonomy_counts": taxonomy_counts,
        "diversity": diversity,
        "contract_version": raw["format"],
        "contract_warnings": []
        if v2
        else ["Legacy v1 contract: use v2 before phase-2 training"],
        "cases": [c.model_dump() for c in cases],
    }
