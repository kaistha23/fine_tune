import json
from datetime import date

import pytest

from credit_risk.dataset import build_dataset, build_sft_record, stable_split, verify_manifest
from credit_risk.training import iterations_for


def payload(group="G", asof="2025-01-01"):
    case = {
        "case_id": "SYNTH-" + group + "-" + asof,
        "obligor_id": "SYNTH-" + group,
        "group_id": group,
        "portfolio": "retail",
        "jurisdiction": "SAMA",
        "as_of_date": asof,
        "current_position": {"stage": 1},
    }
    target = {
        "answer_status": "ANSWERED",
        "executive_summary": "current_position.stage = 1.",
        "facts": [{"statement": "current_position.stage = 1.", "evidence_ids": [case["case_id"]]}],
    }
    return {
        "case": case,
        "target": json.dumps(target),
        "task_type": "factsheet",
        "situation": "base",
        "template_family": "fixture-" + stable_split(group),
        "question": "What stage?",
        "evidence": [],
        "review": {"status": "approved", "reviewer_id": "synthetic-reviewer", "quality_score": 5},
        "data_classification": "synthetic",
    }


def test_snapshot_ids_and_group_holdout(tmp_path):
    a = payload("shared")
    b = payload("shared", "2026-02-01")
    assert (
        build_sft_record(a["case"], a["target"], "factsheet")["example_id"]
        != build_sft_record(b["case"], b["target"], "factsheet")["example_id"]
    )
    m = build_dataset([a, b], tmp_path / "v1", date(2026, 1, 1))
    assert m["counts"] == {"train": 0, "valid": 0, "test": 2}
    assert verify_manifest(tmp_path / "v1")["manifest_hash"] == m["manifest_hash"]
    with pytest.raises(ValueError):
        build_dataset([a], tmp_path / "v1", date(2026, 1, 1))
    (tmp_path / "v1" / "test.jsonl").write_text("changed")
    with pytest.raises(ValueError):
        verify_manifest(tmp_path / "v1")


def test_frozen_exclusions_and_approval(tmp_path):
    a = payload()
    a["review"]["status"] = "pending"
    with pytest.raises(ValueError):
        build_dataset([a], tmp_path / "v1", date(2026, 1, 1))
    a = payload()
    with pytest.raises(ValueError):
        build_dataset(
            [a], tmp_path / "v1", date(2026, 1, 1), {"groups": ["G"], "content_hashes": []}
        )


def test_missing_evidence_target_rejected():
    a = payload()
    a["target"] = json.dumps(
        {
            "answer_status": "ANSWERED",
            "executive_summary": "Invented policy",
            "facts": [{"statement": "Invented policy", "evidence_ids": ["MISSING"]}],
        }
    )
    with pytest.raises(ValueError):
        build_sft_record(a["case"], a["target"], "policy_qa")


def test_iterations_are_micro_batches():
    assert iterations_for(600, 1, 8, 2) == 1200
    assert iterations_for(100, 1, 8, 2) == 200
    assert iterations_for(101, 1, 8, 2) == 208


def test_cli_preserves_question_and_evidence(tmp_path):
    import subprocess
    import sys

    a = payload()
    a["evidence"] = [
        {
            "evidence_id": "SYNTH@1#1",
            "jurisdiction": "SAMA",
            "document_id": "SYNTH",
            "document_version": "1",
            "section": "1",
            "text": "Synthetic guidance.",
            "score": 1,
        }
    ]
    source = tmp_path / "in.jsonl"
    source.write_text(json.dumps(a) + "\n")
    exclusion = tmp_path / "exclude.json"
    exclusion.write_text('{"groups":[],"content_hashes":[]}')
    subprocess.run(
        [
            sys.executable,
            "-m",
            "credit_risk.dataset",
            str(source),
            str(tmp_path / "out"),
            "--out-of-time-from",
            "2026-01-01",
            "--exclusions",
            str(exclusion),
        ],
        check=True,
        capture_output=True,
    )
    messages = json.loads((tmp_path / "out" / (stable_split("G") + ".jsonl")).read_text())[
        "messages"
    ]
    actual = json.loads(messages[1]["content"])
    assert actual["question"] == a["question"]
    assert actual["context"]["evidence"] == a["evidence"]
