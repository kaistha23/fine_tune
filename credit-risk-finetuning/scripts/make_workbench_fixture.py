#!/usr/bin/env python3
"""Create a registrable credit-workbench-v2 dataset from the local synthetic DuckDB."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

from credit_risk.factsheet import build_factsheet
from credit_risk.schemas import CreditResponse, QueryPlan, SupportedClaim
from credit_risk.workbench.contracts import inspect_dataset

SPLIT_TEMPLATES = {
    "train": "Report the current credit stage from the supplied factsheet.",
    "validation": "State the obligor's current stage using the supplied factsheet.",
    "test": "Identify the current stage and cite the supplied factsheet.",
    "oot": "Give the current stage visible at the stated as-of date.",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows_for(con, obligor_id: str, as_of: date) -> list[dict]:
    cursor = con.execute(
        """
        SELECT * FROM obligor_monthly
        WHERE obligor_id = ?
          AND observation_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
          AND data_cutoff_date <= ?
          AND model_run_date <= ?
        ORDER BY observation_date
        """,
        [obligor_id, as_of, as_of],
    )
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def make_case(con, identity: tuple[str, str, str], split: str, source_hash: str) -> dict:
    obligor_id, portfolio, jurisdiction = identity
    as_of = date(2026, 1, 15) if split == "oot" else date(2025, 12, 31)
    plan = QueryPlan(
        portfolio=portfolio,
        jurisdiction=jurisdiction,
        obligor_id=obligor_id,
        date_from=date(2025, 1, 1),
        date_to=date(2025, 12, 31),
        as_of_date=as_of,
        metrics=["current_ratio", "utilisation_pct", "pit_pd", "stage"],
        analysis_type="factsheet",
    )
    case_id = f"LOCAL-{split.upper()}-{obligor_id}-{as_of.isoformat()}"
    factsheet = build_factsheet(rows_for(con, obligor_id, as_of), plan, case_id=case_id)
    complete_facts = factsheet.model_dump(mode="json")
    stage = complete_facts["current_position"]["stage"]
    facts = {
        "schema_version": complete_facts["schema_version"],
        "case_id": case_id,
        "obligor_id": obligor_id,
        "group_id": obligor_id,
        "portfolio": portfolio,
        "jurisdiction": jurisdiction,
        "as_of_date": as_of.isoformat(),
        "current_position": {"stage": stage},
    }
    statement = f"current_position.stage = {stage}"
    target = CreditResponse(
        answer_status="ANSWERED",
        executive_summary=statement,
        facts=[SupportedClaim(statement=statement, evidence_ids=[case_id])],
    ).model_dump(mode="json")
    latest_date = max(row["observation_date"] for row in rows_for(con, obligor_id, as_of))
    fact_records = [
        {
            "fact_id": f"{case_id}-stage",
            "metric": "stage",
            "value": stage,
            "unit": None,
            "currency": None,
            "effective_date": latest_date.isoformat(),
            "source_id": case_id,
        },
    ]
    result = {
        "case_id": case_id,
        "group_id": obligor_id,
        "task": "credit_analysis",
        "task_type": "factsheet",
        "situation": "base",
        "split": split,
        "question": SPLIT_TEMPLATES[split],
        "portfolio": portfolio,
        "jurisdiction": jurisdiction,
        "as_of_date": as_of.isoformat(),
        "facts": facts,
        "fact_records": fact_records,
        "evidence": [],
        "expected": {
            "fields": {"answer_status": "ANSWERED", "facts.0.statement": statement},
            "evidence_ids": [case_id],
            "risk_drivers": [],
            "severity_weight": 1,
        },
        "consistency_paths": ["answer_status", "facts.0.statement"],
        "provenance": {
            "classification": "synthetic",
            "source_snapshot_hash": source_hash,
            "template_family": f"local-stage-{split}-v1",
            "transformations": [
                "deterministic-local-fixture",
                "factsheet-from-governed-calculators",
            ],
        },
    }
    if split in ("train", "validation"):
        result["target"] = target
    return result


def build(source: Path, output: Path, version: str, counts: dict[str, int]) -> dict:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    total = sum(counts.values())
    with duckdb.connect(str(source), read_only=True) as con:
        identities = con.execute(
            """
            SELECT DISTINCT obligor_id, portfolio, jurisdiction
            FROM obligor_monthly ORDER BY obligor_id LIMIT ?
            """,
            [total],
        ).fetchall()
        if len(identities) != total:
            raise ValueError(f"Need {total} distinct obligors; source has {len(identities)}")
        source_hash = sha256(source)
        output.mkdir(parents=True)
        offset = 0
        splits = {}
        try:
            for split, count in counts.items():
                selected = identities[offset : offset + count]
                offset += count
                path = output / f"{split}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps(make_case(con, row, split, source_hash), allow_nan=False) + "\n"
                        for row in selected
                    )
                )
                splits[split] = {"file": path.name, "sha256": sha256(path)}
            manifest = {
                "format": "credit-workbench-v2",
                "dataset_id": "local-synthetic-credit-analysis",
                "dataset_version": version,
                "created_at": datetime.now(UTC).isoformat(),
                "name": f"Local synthetic credit analysis {version}",
                "task": "credit_analysis",
                "oot_from": "2026-01-01",
                "splits": splits,
            }
            manifest_path = output / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            inspected = inspect_dataset(manifest_path)
        except Exception:
            shutil.rmtree(output)
            raise
    return {"manifest": str(manifest_path), "counts": inspected["counts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/curated/credit_risk.duckdb"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--oot", type=int, default=16)
    args = parser.parse_args()
    counts = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
        "oot": args.oot,
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("Every split count must be positive")
    print(json.dumps(build(args.source, args.out, args.version, counts), indent=2))


if __name__ == "__main__":
    main()
