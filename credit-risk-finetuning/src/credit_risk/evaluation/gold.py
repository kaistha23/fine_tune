"""Load and freeze the gold evaluation set.

The scoring, gates and champion-challenger comparison were built with no cases to consume.
This is the loader, plus the freeze that makes a regression suite meaningful.

A gold set that quietly changes between runs cannot detect a regression: the comparison
would attribute the difference to the adapter. So the file carries a content hash, and
loading verifies it. Editing a case is allowed - silently editing one is not.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from credit_risk.evaluation.metrics import GoldCase
from credit_risk.schemas import AnswerStatus, Evidence


class GoldSetError(ValueError):
    pass


def content_hash(path: Path) -> str:
    """Hash of the case payloads, ignoring the manifest line that records it."""
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    payloads = []
    for line in lines:
        record = json.loads(line)
        if record.get("record_type") == "manifest":
            continue
        payloads.append(json.dumps(record, sort_keys=True))
    return hashlib.sha256("\n".join(payloads).encode("utf-8")).hexdigest()


def _to_case(record: dict) -> GoldCase:
    return GoldCase(
        case_id=record["case_id"],
        portfolio=record["portfolio"],
        task_type=record["task_type"],
        jurisdiction=record["jurisdiction"],
        question=record["question"],
        available_evidence=[Evidence(**item) for item in record.get("available_evidence", [])],
        expected_status=AnswerStatus(record.get("expected_status", "ANSWERED")),
        required_evidence_ids=record.get("required_evidence_ids", []),
        required_risk_drivers=record.get("required_risk_drivers", []),
        expected_numerics=record.get("expected_numerics", {}),
        must_abstain=record.get("must_abstain", False),
        is_injection_attempt=record.get("is_injection_attempt", False),
    )


def load_gold_set(path: str | Path, verify_hash: bool = True) -> list[GoldCase]:
    path = Path(path)
    if not path.is_file():
        raise GoldSetError(f"Gold set not found: {path.name}")

    manifest: dict | None = None
    cases: list[GoldCase] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        record = json.loads(line)
        if record.get("record_type") == "manifest":
            manifest = record
            continue
        if record["case_id"] in seen:
            raise GoldSetError(f"Duplicate gold case id: {record['case_id']}")
        seen.add(record["case_id"])
        cases.append(_to_case(record))

    if verify_hash:
        if manifest is None or "content_hash" not in manifest:
            raise GoldSetError("Gold set has no manifest recording its content hash")
        observed = content_hash(path)
        if observed != manifest["content_hash"]:
            raise GoldSetError(
                "Gold set has changed since it was frozen. A regression suite whose cases "
                "move cannot attribute a difference to the adapter. Re-freeze deliberately "
                f"if the change is intended (expected {manifest['content_hash'][:12]}, "
                f"found {observed[:12]})."
            )
    return cases


def coverage(cases: list[GoldCase]) -> dict:
    """Where the set is thin. An untested slice cannot gate a release."""
    portfolios: dict[str, int] = {}
    tasks: dict[str, int] = {}
    for case in cases:
        portfolios[case.portfolio] = portfolios.get(case.portfolio, 0) + 1
        tasks[case.task_type] = tasks.get(case.task_type, 0) + 1
    return {
        "cases": len(cases),
        "by_portfolio": dict(sorted(portfolios.items())),
        "by_task": dict(sorted(tasks.items())),
        "abstention_cases": sum(1 for c in cases if c.must_abstain),
        "injection_cases": sum(1 for c in cases if c.is_injection_attempt),
        "uncovered_portfolios": sorted(
            {"retail", "sme", "corporate"} - set(portfolios)),
    }
