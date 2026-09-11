"""Reviewed, group/time isolated SFT datasets with immutable manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from credit_risk.guardrails import validate_output
from credit_risk.prompts import PROMPT_VERSION, build_messages
from credit_risk.review_store import digest
from credit_risk.schemas import CreditResponse, Evidence

DEFAULT_QUESTION = "Assess the supplied credit factsheet using the supplied evidence."
DATASET_VERSION = "v3.0.0"
SPLITS = ("train", "valid", "test")


def stable_split(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "valid" if bucket < 85 else "test"


def build_sft_record(
    case: dict[str, Any],
    target: str,
    task_type: str,
    question: str = "",
    evidence: list[dict[str, Any]] | None = None,
) -> dict:
    evidence = evidence or []
    response = CreditResponse.model_validate_json(target)
    check = validate_output(
        response, [Evidence.model_validate(e) for e in evidence], case.get("case_id"), case
    )
    if not check.passed:
        raise ValueError("Target failed evidence validation: " + ",".join(check.failures))
    group = str(case.get("group_id") or case["obligor_id"])
    return {
        "example_id": "SFT-"
        + digest(
            {
                "case": case,
                "task": task_type,
                "question": question,
                "evidence": evidence,
                "target": target,
            }
        )[:24],
        "group_id": group,
        "obligor_id": str(case["obligor_id"]),
        "portfolio": case["portfolio"],
        "task_type": task_type,
        "jurisdiction": case["jurisdiction"],
        "as_of_date": case["as_of_date"],
        "messages": [
            *build_messages(question or DEFAULT_QUESTION, case, evidence),
            {"role": "assistant", "content": target},
        ],
        "split": stable_split(group),
        "dataset_version": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
    }


def to_chat_line(record):
    return json.dumps({"messages": record["messages"]}, ensure_ascii=False)


def to_provenance_line(record):
    return json.dumps({k: v for k, v in record.items() if k != "messages"}, ensure_ascii=False)


def build_dataset(payloads, output: Path, cutoff: date, exclusions: dict | None = None):
    if output.exists():
        raise ValueError("Dataset output must be a new version directory")
    exclusions = exclusions or {"groups": [], "content_hashes": []}
    blocked_groups = set(exclusions["groups"])
    blocked_hashes = set(exclusions["content_hashes"])
    records = []
    seen = set()
    ownership = {}
    future = set()
    for payload in payloads:
        review = payload.get("review", {})
        if (
            review.get("status") != "approved"
            or not review.get("reviewer_id")
            or review.get("quality_score", 0) < 4
        ):
            raise ValueError("Named SME approval and quality score >=4 required")
        if payload.get("data_classification") not in ("synthetic", "masked"):
            raise ValueError("Only synthetic or masked records accepted")
        case = payload["case"]
        if not case.get("group_id"):
            raise ValueError("Explicit group_id required")
        group = case["group_id"]
        obligor = case["obligor_id"]
        if obligor in ownership and ownership[obligor] != group:
            raise ValueError("Conflicting group lineage")
        ownership[obligor] = group
        record = build_sft_record(
            case,
            payload["target"],
            payload["task_type"],
            payload.get("question", ""),
            payload.get("evidence", []),
        )
        content = digest(record["messages"])
        if content in seen:
            continue
        seen.add(content)
        record.update(
            review=review, data_classification=payload["data_classification"], content_hash=content
        )
        if date.fromisoformat(case["as_of_date"]) >= cutoff:
            future.add(group)
        records.append(record)
    # All cases belonging to a future/test group stay out of train/validation, including
    # its earlier snapshots. Existing gold groups and exact content are never emitted.
    frozen_groups = future | {r["group_id"] for r in records if r["split"] == "test"}
    for r in records:
        if r["group_id"] in frozen_groups:
            r["split"] = "test"
    records = [
        r
        for r in records
        if r["group_id"] not in blocked_groups and r["content_hash"] not in blocked_hashes
    ]
    if not records:
        raise ValueError("No eligible records")
    output.mkdir(parents=True)
    hashes = {}
    counts = {}
    for split in SPLITS:
        selected = [r for r in records if r["split"] == split]
        for suffix, serialise in [
            (".jsonl", to_chat_line),
            (".provenance.jsonl", to_provenance_line),
        ]:
            path = output / (split + suffix)
            path.write_text("".join(serialise(r) + "\n" for r in selected))
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[split] = len(selected)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
        "out_of_time_from": cutoff.isoformat(),
        "files": hashes,
        "counts": counts,
        "group_assignments": {r["group_id"]: r["split"] for r in records},
        "excluded_groups": sorted(blocked_groups),
        "excluded_content_hashes": sorted(blocked_hashes),
        "gold_exclusions": {
            "groups": sorted(frozen_groups | blocked_groups),
            "content_hashes": sorted(
                blocked_hashes | {r["content_hash"] for r in records if r["split"] == "test"}
            ),
        },
    }
    manifest["manifest_hash"] = digest(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_manifest(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    expected = manifest.pop("manifest_hash")
    if digest(manifest) != expected:
        raise ValueError("Manifest changed")
    for name, expected_file in manifest["files"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((path / name).read_bytes()).hexdigest() != expected_file
        ):
            raise ValueError("Dataset file changed")
    manifest["manifest_hash"] = expected
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--out-of-time-from", type=date.fromisoformat, required=True)
    p.add_argument(
        "--exclusions",
        type=Path,
        required=True,
        help="Frozen gold exclusion JSON; explicit empty lists allowed for initial synthetic dataset",
    )
    a = p.parse_args()
    payloads = [json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    print(
        json.dumps(
            build_dataset(
                payloads, a.output_dir, a.out_of_time_from, json.loads(a.exclusions.read_text())
            )["counts"]
        )
    )


if __name__ == "__main__":
    main()
