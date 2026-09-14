"""Continuous-learning steps: verified-answer replay, consistency gates, dataset builder.

Replay re-runs every verified answer in its frozen context (the stored case: factsheet,
evidence, rules and prompt/schema version) against a candidate model, so "still answers the
same after retraining" is measured instead of assumed. The dataset builder turns eligible
feedback into a new immutable V2 dataset version next to curated anchors; registering and
training remain explicit steps.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from credit_risk.review_store import digest
from credit_risk.workbench import memory
from credit_risk.workbench.contracts import SPLITS, Case, inspect_dataset, messages
from credit_risk.workbench.evaluation import assess
from credit_risk.workbench.feedback import batch_records

V2_LINEAGE = ("source_snapshot_hash", "template_family", "transformations")


class LearningError(ValueError):
    pass


# -- replay ---------------------------------------------------------------------------------
def verified_answers(store, task: str = "credit_analysis") -> list[dict]:
    records = [
        record
        for record in memory.latest(store, status="verified")
        if store.get("answer", record["answer_id"])["case"]["task"] == task
    ]
    return sorted(records, key=lambda record: record["created_at"])


def replay(store, answer_ids: list[str], generate, progress=None) -> dict:
    """Generate once per verified answer (greedy, first seed) and compare decision fields."""
    mismatches = []
    matched = 0
    for number, answer_id in enumerate(answer_ids, start=1):
        answer = store.get("answer", answer_id)
        record = memory.latest(store, answer_id=answer_id)[-1]
        case = Case.model_validate(answer["case"])
        version = store.get("version", answer["version_id"])
        text = generate(messages(case, version), 42)
        _metrics, failures, parsed = assess(case, text, version)
        fields = memory.consistency_fields(parsed["answer"]) if parsed else None
        differences = memory.field_diff(record["fields"], fields) if fields else {"output": "invalid"}
        if differences:
            mismatches.append(
                {
                    "answer_id": answer_id,
                    "question": answer["case"]["question"],
                    "differences": differences,
                    "failures": failures,
                    "output": text if isinstance(text, str) else json.dumps(text),
                }
            )
        else:
            matched += 1
        if progress:
            progress(number, len(answer_ids))
    return {
        "kind": "replay",
        "total": len(answer_ids),
        "matched": matched,
        "mismatches": mismatches,
        "passed": bool(answer_ids) and matched == len(answer_ids),
        "note": "Frozen verified contexts; decision fields compared, not prose.",
    }


# -- consistency gates ----------------------------------------------------------------------
def load_gates(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data.get("gates"), dict) or not data["gates"]:
        raise LearningError("Consistency gate file declares no gates")
    return data


def consistency_gates(store, gates: dict, candidate: dict, training_job_id: str, checkpoint_sha: str | None) -> dict:
    """Candidate evaluation metrics, verified replay and regression checks against thresholds."""
    results = []
    splits = (candidate.get("result") or {}).get("splits", {})
    for metric in ("repeated_agreement", "equivalent_agreement", "negative_control_distinction"):
        rule = gates["gates"].get(metric)
        if not rule:
            continue
        for split in ("validation", "test", "oot"):
            measured = splits.get(split, {}).get(metric)
            if measured is None:
                results.append({"gate": metric, "scope": split, "status": "not_evaluated", "minimum": rule["minimum"]})
                continue
            results.append(
                {
                    "gate": metric,
                    "scope": split,
                    "value": measured["value"],
                    "denominator": measured["denominator"],
                    "minimum": rule["minimum"],
                    "status": "passed" if measured["value"] >= rule["minimum"] else "failed",
                }
            )
    jobs = store.list("job")
    rule = gates["gates"].get("verified_replay")
    if rule:
        replays = [
            job
            for job in jobs
            if job["spec"].get("kind") == "replay"
            and job["status"] == "completed"
            and job["spec"].get("adapter_job_id") == training_job_id
            and job["spec"].get("checkpoint_sha256") == checkpoint_sha
        ]
        result = _result(replays[-1]) if replays else None
        if not result or not result.get("total"):
            results.append({"gate": "verified_replay", "scope": "verified answers", "status": "not_evaluated", "minimum": rule["minimum"]})
        else:
            value = result["matched"] / result["total"]
            results.append(
                {
                    "gate": "verified_replay",
                    "scope": "verified answers",
                    "value": value,
                    "denominator": result["total"],
                    "minimum": rule["minimum"],
                    "status": "passed" if value >= rule["minimum"] else "failed",
                    "job_id": replays[-1]["id"],
                }
            )
    rule = gates["gates"].get("regression_checks")
    if rule:
        task = candidate["job"]["spec"]["task"]
        regressions = [j for j in jobs if j["spec"].get("kind") == "regression" and j["status"] == "completed" and j["spec"]["task"] == task]
        mine = [j for j in regressions if j["spec"].get("adapter_job_id") == training_job_id]
        reference = [j for j in regressions if not j["spec"].get("adapter_job_id")]
        if not mine or not reference:
            results.append({"gate": "regression_checks", "scope": "development checks", "status": "not_evaluated", "note": "Run Feedback regression for the base model and this adapter"})
        else:
            before = _passing(_result(reference[-1]))
            after = _passing(_result(mine[-1]))
            lost = sorted(before - after)
            results.append(
                {
                    "gate": "regression_checks",
                    "scope": "development checks",
                    "value": len(lost),
                    "maximum_newly_failing": rule.get("maximum_newly_failing", 0),
                    "newly_failing": lost,
                    "status": "passed" if len(lost) <= rule.get("maximum_newly_failing", 0) else "failed",
                }
            )
    statuses = {item["status"] for item in results}
    return {
        "version": gates.get("version"),
        "results": results,
        "status": "failed" if "failed" in statuses else ("not_evaluated" if "not_evaluated" in statuses else "passed"),
    }


def _result(job):
    path = Path(job["spec"]["output"]) / "result.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _passing(result):
    return {row["case_id"] for row in (result or {}).get("cases", []) if not row["failures"]}


# -- dataset builder ------------------------------------------------------------------------
def build_dataset(
    store,
    base_dataset: dict,
    dataset_version: str,
    output_root: Path,
    paraphrases: dict[str, list[str]] | None = None,
    feedback_fraction: float = 0.2,
) -> dict:
    """Base anchors + eligible feedback (≤ feedback_fraction of train) as a new V2 version."""
    manifest = base_dataset["manifest"]
    if manifest.get("format") != "credit-workbench-v2":
        raise LearningError("Build from a credit-workbench-v2 dataset")
    if not 0 < feedback_fraction < 1:
        raise LearningError("Feedback fraction must be between 0 and 1")
    task = manifest["task"]
    source = Path(base_dataset["path"]).parent
    target = Path(output_root) / task.replace("_", "-") / dataset_version
    if target.exists():
        raise LearningError(f"Dataset version already exists: {target}")
    fresh = inspect_dataset(base_dataset["path"])
    if fresh["hash"] != base_dataset["hash"]:
        raise LearningError("Base dataset changed after registration")
    base_cases = fresh["cases"]
    oot_from = date.fromisoformat(manifest["oot_from"])
    base_ids = {case["case_id"] for case in base_cases}
    base_fingerprints = {digest({"q": c["question"].strip().casefold(), "c": Case.model_validate(c).context_hash()}) for c in base_cases}
    batch = batch_records(store, task)
    skipped = dict(batch.get("skipped_reasons", {}))
    added, seen_ids = [], set()
    limit = int(feedback_fraction / (1 - feedback_fraction) * fresh["counts"]["train"])
    paraphrases = paraphrases or {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for case in reversed(batch["cases"]):  # latest feedback first
        feedback_id = case["provenance"].get("feedback_id")
        if any(key not in case["provenance"] for key in V2_LINEAGE):
            skip("missing_v2_lineage")
            continue
        if date.fromisoformat(case["as_of_date"]) >= oot_from:
            skip("after_oot_cutoff")
            continue
        if case["case_id"] in base_ids or case["case_id"] in seen_ids:
            skip("duplicate_case_id")
            continue
        fingerprint = digest({"q": case["question"].strip().casefold(), "c": Case.model_validate(case).context_hash()})
        if fingerprint in base_fingerprints:
            skip("duplicates_base_case")
            continue
        family = [case]
        for index, text in enumerate(paraphrases.get(feedback_id, []), start=1):
            question = " ".join(str(text).split())
            if question and question.casefold() != case["question"].strip().casefold():
                family.append({**case, "case_id": f"{case['case_id']}-p{index}", "question": question})
        if len(family) > 1:
            for member in family:
                member["equivalence_id"] = "feedback-" + feedback_id
                member["provenance"] = {**member["provenance"], "paraphrase_family": feedback_id}
        if len(added) + len(family) > limit:
            skip("feedback_fraction_cap")
            continue
        seen_ids.update(member["case_id"] for member in family)
        base_fingerprints.update(
            digest({"q": m["question"].strip().casefold(), "c": Case.model_validate(m).context_hash()}) for m in family
        )
        added.extend(family)
    if not added:
        raise LearningError("No eligible feedback can be added: " + json.dumps(skipped))
    target.mkdir(parents=True)
    try:
        splits = {}
        for split in SPLITS:
            entry = manifest["splits"][split]
            content = (source / entry["file"]).read_bytes()
            if split == "train":
                content += "".join(json.dumps(case, allow_nan=False) + "\n" for case in added).encode()
            (target / f"{split}.jsonl").write_bytes(content)
            splits[split] = {"file": f"{split}.jsonl", "sha256": hashlib.sha256(content).hexdigest()}
        new_manifest = {
            **manifest,
            "dataset_version": dataset_version,
            "created_at": datetime.now(UTC).isoformat(),
            "name": f"{manifest.get('name', manifest['dataset_id'])} + feedback {dataset_version}",
            "splits": splits,
            "derived_from": {
                "dataset_hash": base_dataset["hash"],
                "dataset_version": manifest["dataset_version"],
                "feedback_ids": sorted({case["provenance"]["feedback_id"] for case in added}),
                "feedback_batch_hash": batch["hash"],
                "feedback_fraction_limit": feedback_fraction,
            },
        }
        registry = manifest.get("schema_registry")
        if registry:
            shutil.copyfile(source / registry["file"], target / registry["file"])
        snapshot = manifest.get("synthetic_snapshot")
        if snapshot:
            shutil.copyfile(source / snapshot["file"], target / snapshot["file"])
        path = target / "manifest.json"
        path.write_text(json.dumps(new_manifest, indent=2) + "\n")
        inspected = inspect_dataset(path)
    except Exception as exc:
        shutil.rmtree(target, ignore_errors=True)
        if isinstance(exc, LearningError):
            raise
        raise LearningError("Built dataset failed validation: " + str(exc)) from exc
    return {
        "manifest_path": str(path),
        "counts": inspected["counts"],
        "added_cases": len(added),
        "feedback_ids": new_manifest["derived_from"]["feedback_ids"],
        "paraphrase_cases": sum(1 for case in added if case["case_id"].rsplit("-p", 1)[-1].isdigit() and "-p" in case["case_id"]),
        "skipped": skipped,
        "note": "Not registered. Register it in Datasets, then preflight and train explicitly.",
    }
