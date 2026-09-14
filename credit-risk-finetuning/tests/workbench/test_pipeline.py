import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from credit_risk.workbench.contracts import Case, default_version, inspect_dataset, messages
from credit_risk.workbench.evaluation import compare, evaluate
from credit_risk.workbench.feedback import batch_records, submit
from credit_risk.workbench.jobs import Jobs
from credit_risk.workbench.server import Config, create_app
from credit_risk.workbench.store import Store


def case(**updates):
    payload = {
        "case_id": "fixture-1",
        "group_id": "fixture-group",
        "task": "credit_analysis",
        "split": "development",
        "question": "Is enough information available?",
        "portfolio": "retail",
        "jurisdiction": "SAMA",
        "as_of_date": "2025-01-01",
        "facts": {"case_id": "fixture-1"},
        "expected": {"fields": {"answer_status": "INSUFFICIENT_EVIDENCE"}, "must_abstain": True},
        "consistency_paths": ["answer_status", "risk_drivers"],
        "provenance": {"classification": "synthetic"},
    }
    payload.update(updates)
    return Case(**payload)


def answer(status="INSUFFICIENT_EVIDENCE", summary=""):
    return {"answer_status": status, "executive_summary": summary, "risk_drivers": []}


@pytest.fixture
def setup(tmp_path):
    app = create_app(tmp_path, False)
    client = TestClient(app)
    client.headers["X-Workbench-Token"] = client.get("/api/session").json()["token"]
    store = app.state.store
    version = store.get("version", store.active_version("credit_analysis"))
    record = store.add(
        "answer",
        {
            "case": case().model_dump(),
            "output": answer("ANSWERED"),
            "version_id": version["id"],
            "identity": {"model": "fixture-only"},
            "sql_lineage": None,
        },
    )
    return client, store, version, record


def manifest(root, cases):
    root.mkdir(exist_ok=True)
    m = {
        "format": "credit-workbench-v1",
        "name": "Unit fixtures",
        "task": "credit_analysis",
        "oot_from": "2026-01-01",
        "splits": {},
    }
    for split in ("train", "validation", "test", "oot"):
        file = root / (split + ".jsonl")
        file.write_text("".join(c.model_dump_json() + "\n" for c in cases if c.split == split))
        m["splits"][split] = {
            "file": file.name,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
    path = root / "manifest.json"
    path.write_text(json.dumps(m))
    return path


def test_empty_ui_and_origin_boundary(setup):
    c, _, _, _ = setup
    assert c.get("/").status_code == 200
    assert "Not evaluated" in c.get("/").text
    assert c.get("/api/state").json()["jobs"] == []
    assert (
        c.post(
            "/api/datasets", json={"path": "missing"}, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert (
        c.post(
            "/api/datasets", json={"path": "missing"}, headers={"X-Workbench-Token": ""}
        ).status_code
        == 403
    )
    assert c.get("/api/session", headers={"Host": "evil.example"}).status_code == 403


def test_submit_comment_without_workflow(setup):
    c, s, v, a = setup
    r = c.post(
        "/api/feedback",
        json={"submission_id": "f1", "interaction_id": a["id"], "comment": "Wrong conclusion"},
    )
    assert r.status_code == 200
    assert r.json()["regression_status"] == "needs_expected_results"
    assert not r.json()["eligible_for_training"]
    assert s.active_version("credit_analysis") == v["id"]
    assert s.list("job") == []


def test_correction_creates_regression_and_latest_wins(setup):
    c, s, v, a = setup
    body = {
        "submission_id": "f2",
        "interaction_id": a["id"],
        "comment": "Insufficient data",
        "correction": answer(),
        "cause": "model_behaviour",
    }
    first = c.post("/api/feedback", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["eligible_for_training"]
    assert c.post("/api/feedback", json=body).json()["id"] == first.json()["id"]
    assert len(s.list("feedback")) == 1 and len(s.list("regression")) == 1
    assert len(batch_records(s, "credit_analysis")["cases"]) == 1
    body.update(submission_id="f3", correction="{bad json")
    bad = c.post("/api/feedback", json=body)
    assert bad.status_code == 200 and bad.json()["diagnostics"]
    assert len(s.list("feedback")) == 2
    assert batch_records(s, "credit_analysis")["cases"] == []
    assert s.active_version("credit_analysis") == v["id"]


def test_conflicting_retry_fails(setup):
    c, _s, _v, a = setup
    body = {"submission_id": "idempotent", "interaction_id": a["id"], "comment": "one"}
    assert c.post("/api/feedback", json=body).status_code == 200
    assert c.post("/api/feedback", json={**body, "comment": "two"}).status_code == 422


@pytest.mark.parametrize("split", ["validation", "test", "oot"])
def test_frozen_feedback_never_trainable(setup, split):
    _c, s, _v, a = setup
    a = s.add(
        "answer",
        {
            **{k: v for k, v in a.items() if k not in ("id", "created_at")},
            "case": case(split=split).model_dump(),
        },
    )
    r = submit(s, a["id"], "frozen", "wrong", answer())
    assert not r["eligible_for_training"] and r["protected_split_or_group"]
    assert r["regression_status"] == "ready"


def test_dataset_checksum_and_split_isolation(tmp_path):
    p = manifest(tmp_path, [case(split="train")])
    result = inspect_dataset(p)
    assert result["counts"] == {"train": 1, "validation": 0, "test": 0, "oot": 0}
    assert inspect_dataset(tmp_path)["path"] == str(p.resolve())
    (tmp_path / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        inspect_dataset(p)
    p = manifest(
        tmp_path, [case(split="train"), case(case_id="other", split="test", question="other")]
    )
    with pytest.raises(ValueError, match="Group leakage"):
        inspect_dataset(p)


def test_oot_contract(tmp_path):
    p = manifest(tmp_path, [case(split="oot")])
    with pytest.raises(ValueError, match="precedes"):
        inspect_dataset(p)


def test_consistency_repeats_and_paraphrases():
    v = default_version("credit_analysis")
    a = case(equivalence_id="equivalent")
    b = case(case_id="b", question="Do we have sufficient inputs?", equivalence_id="equivalent")
    output = evaluate(
        [a, b], lambda c, v: answer(), v, {"version_id": "v", "generation": {"temperature": 0}}
    )
    metrics = output["splits"]["development"]
    assert metrics["repeated_agreement"]["value"] == 1.0
    assert metrics["repeated_agreement"]["denominator"] == 2
    assert metrics["repeated_agreement"]["ci95"] == [1.0, 1.0]
    assert metrics["equivalent_agreement"]["value"] == 1
    assert output["splits"]["oot"] == {}


def test_inconsistent_and_changed_question_detected():
    calls = iter([answer(), answer("ANSWERED"), answer()])
    v = default_version("credit_analysis")
    report = evaluate([case()], lambda c, v: next(calls), v, {})
    assert report["splits"]["development"]["repeated_agreement"]["value"] == 0
    a = case(distinct_from=["b"])
    b = case(case_id="b", question="Changed borrower?", facts={"obligor_id": "OTHER"})
    report = evaluate([a, b], lambda c, v: answer(), v, {})
    assert report["splits"]["development"]["negative_control_distinction"]["value"] == 0


def test_same_wording_cannot_mask_wrong_numbers():
    v = default_version("credit_analysis")
    c = case(expected={"numerics": {"value": 2}}, consistency_paths=["value"])
    report = evaluate([c], lambda c, v: {**answer(), "value": 99}, v, {})
    assert report["splits"]["development"]["numerical_agreement"]["value"] == 0


def test_comparisons_reject_changed_inputs_and_versions():
    v = default_version("credit_analysis")
    a = evaluate([case()], lambda c, v: answer(), v, {"version_id": "v", "generation": {}})
    b = {**a, "case_set_hash": "different"}
    with pytest.raises(ValueError):
        compare(a, b)
    b = {**a, "identity": {"version_id": "other", "generation": {}}}
    with pytest.raises(ValueError):
        compare(a, b)


def test_save_version_and_rollback_without_approval(setup):
    c, s, v, a = setup
    body = {k: v[k] for k in ("task", "prompt", "schema", "name")}
    body.update(parent=v["id"], name="new", prompt=v["prompt"] + " Check missing inputs.")
    new = c.post("/api/versions", json=body).json()
    assert s.active_version("credit_analysis") == v["id"]
    assert c.post("/api/versions/" + new["id"] + "/activate", json={}).status_code == 200
    assert s.active_version("credit_analysis") == new["id"]
    c.post("/api/versions/" + v["id"] + "/activate", json={})
    assert s.active_version("credit_analysis") == v["id"]
    assert s.get("answer", a["id"])["version_id"] == v["id"]


def test_new_recommended_contract_is_saved_without_implicit_activation(tmp_path):
    store = Store(tmp_path)
    legacy = default_version("credit_analysis")
    for key in (
        "conclusions",
        "risk_driver_details",
        "missing_information_details",
        "recommendation_detail",
    ):
        legacy["schema"]["properties"].pop(key)
    legacy["name"] = "Legacy active contract"
    active = store.add("version", legacy)
    store.active_version("credit_analysis", active["id"])
    create_app(tmp_path, False)
    reloaded = Store(tmp_path)
    assert reloaded.active_version("credit_analysis") == active["id"]
    recommended = [
        version
        for version in reloaded.list("version")
        if version["name"] == "Recommended structured contract v3"
    ]
    assert len(recommended) == 1
    assert "risk_driver_details" in recommended[0]["schema"]["properties"]
    claim = recommended[0]["schema"]["$defs"]["SupportedClaim"]
    assert "derivation" in claim["properties"]


def test_prompt_contains_output_schema():
    v = default_version("credit_analysis")
    assert json.loads(messages(case(), v)[1]["content"])["response_schema"] == v["schema"]
    from credit_risk.prompts import build_messages

    assert "response_schema" in json.loads(build_messages("q", {}, [])[1]["content"])


class FakeProcess:
    pid = 9999999
    code = None

    def poll(self):
        return self.code


def test_single_lane_and_failure(tmp_path):
    s = Store(tmp_path)
    launched = []

    def launch(*args, **kw):
        p = FakeProcess()
        launched.append(p)
        return p

    jobs = Jobs(s, launch)
    first = jobs.enqueue({"kind": "train"})
    second = jobs.enqueue({"kind": "evaluate"})
    jobs.tick()
    jobs.tick()
    assert len(launched) == 1 and s.get("job", second["id"])["status"] == "queued"
    launched[0].code = 1
    jobs.tick()
    assert s.get("job", first["id"])["status"] == "failed" and len(launched) == 2
    launched[1].code = 0
    jobs.tick()
    assert s.get("job", second["id"])["status"] == "completed"


def test_restart_and_queued_cancel(tmp_path):
    s = Store(tmp_path)
    jobs = Jobs(s)
    interrupted = jobs.enqueue({"kind": "train"})
    s.update_job(interrupted["id"], status="running")
    queued = jobs.enqueue({"kind": "evaluate"})
    assert jobs.stop(queued["id"])["status"] == "cancelled"
    jobs.start()
    jobs.close()
    assert s.get("job", interrupted["id"])["status"] == "interrupted"


def test_query_plan_normalization_and_wrong_entity():
    plan = {
        "portfolio": "corporate",
        "jurisdiction": "SAMA",
        "obligor_id": "SYNTH-1",
        "date_from": "2025-01-01",
        "date_to": "2025-01-31",
        "as_of_date": "2025-02-01",
        "metrics": ["pit_pd", "stage"],
    }
    c = case(task="query_plan", expected={"query_plan": plan}, consistency_paths=[])
    v = default_version("query_plan")
    outputs = iter([plan, {**plan, "metrics": ["stage", "pit_pd"]}, plan])
    report = evaluate([c], lambda c, v: next(outputs), v, {})
    assert report["splits"]["development"]["repeated_agreement"]["value"] == 1
    report = evaluate([c], lambda c, v: {**plan, "obligor_id": "WRONG"}, v, {})
    assert report["splits"]["development"]["plan:obligor_id"]["value"] == 0


def test_real_query_compiler_and_synthetic_result_check(tmp_path):
    import duckdb

    from credit_risk.workbench.worker import query_checker

    db = tmp_path / "synthetic.duckdb"
    # Only a schema-shaped test fixture is created; no domain dataset or training runs.
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE obligor_monthly (obligor_id VARCHAR, portfolio VARCHAR, jurisdiction VARCHAR, observation_date DATE, data_cutoff_date DATE, model_run_date DATE, pit_pd DOUBLE)"
    )
    con.execute(
        "INSERT INTO obligor_monthly VALUES ('SYNTH-1','corporate','SAMA','2025-01-31','2025-02-01','2025-02-01',0.02)"
    )
    con.close()
    spec = {
        "dataset": {
            "path": str(tmp_path / "manifest.json"),
            "manifest": {
                "synthetic_snapshot": {
                    "file": db.name,
                    "classification": "synthetic",
                    "sha256": hashlib.sha256(db.read_bytes()).hexdigest(),
                }
            },
        }
    }
    plan = {
        "portfolio": "corporate",
        "jurisdiction": "SAMA",
        "obligor_id": "SYNTH-1",
        "date_from": "2025-01-01",
        "date_to": "2025-01-31",
        "as_of_date": "2025-02-01",
        "metrics": ["pit_pd"],
    }
    checker = query_checker(spec)
    # Compilation path does not execute a DB without expected result rows.
    metrics, packet = checker(case(task="query_plan", expected={}), plan)
    assert metrics["compilation_success"] == 1
    assert "SELECT" in packet["parameterised_sql"]
    assert set(packet["parameter_values"]) == {"***MASKED***"}
    expected = [
        {
            "obligor_id": "SYNTH-1",
            "portfolio": "corporate",
            "jurisdiction": "SAMA",
            "observation_date": "2025-01-31",
            "data_cutoff_date": "2025-02-01",
            "model_run_date": "2025-02-01",
            "pit_pd": 0.02,
        }
    ]
    metrics, packet = checker(case(task="query_plan", expected={"rows": expected}), plan)
    assert metrics["result_agreement"] == 1


def test_comment_then_string_correction_exports_dict(setup):
    _c, s, _v, a = setup
    r = submit(
        s,
        a["id"],
        "string-correction",
        "corrected",
        json.dumps(answer()),
        cause="model_behaviour",
    )
    assert r["eligible_for_training"]
    assert isinstance(batch_records(s, "credit_analysis")["cases"][0]["target"], dict)


def test_component_error_never_enters_training(setup):
    _c, s, _v, a = setup
    r = submit(s, a["id"], "data-error", "wrong database row", answer(), cause="data")
    assert r["correction_valid"] and not r["eligible_for_training"]


def test_state_survives_restart(setup):
    _c, s, v, a = setup
    submit(s, a["id"], "persist", "wrong")
    reloaded = Store(s.root)
    assert reloaded.get("feedback", "persist")["snapshot"]["id"] == a["id"]
    assert reloaded.active_version("credit_analysis") == v["id"]


def test_sql_import_masks_parameter_values(setup):
    c, _s, v, _a = setup
    r = c.post(
        "/api/answers",
        json={
            "case": case().model_dump(),
            "output": answer(),
            "version_id": v["id"],
            "sql_lineage": {
                "parameterised_sql": "SELECT * FROM t WHERE id=?",
                "parameter_values": ["secret-fixture-id"],
            },
        },
    )
    assert r.status_code == 200
    assert "secret-fixture-id" not in r.text


def test_missing_expected_metrics_are_not_perfect_scores():
    result = evaluate(
        [case(expected={}, consistency_paths=[])],
        lambda c, v: answer(),
        default_version("credit_analysis"),
        {},
    )
    metrics = result["splits"]["development"]
    assert "numerical_agreement" not in metrics and "repeated_agreement" not in metrics


def test_equivalent_context_changes_are_rejected(tmp_path):
    a = case(split="test", equivalence_id="e")
    b = case(
        case_id="b",
        group_id="g2",
        split="test",
        question="similar?",
        facts={"borrower": "different"},
        equivalence_id="e",
    )
    with pytest.raises(ValueError, match="Equivalent"):
        inspect_dataset(manifest(tmp_path, [a, b]))


def test_incompatible_schema_saved_but_cannot_activate(setup):
    c, s, v, _a = setup
    body = {
        "task": "credit_analysis",
        "prompt": v["prompt"],
        "schema": {"type": "object"},
        "name": "Incompatible",
        "parent": v["id"],
    }
    saved = c.post("/api/versions", json=body)
    assert saved.status_code == 200
    assert c.post("/api/versions/" + saved.json()["id"] + "/activate", json={}).status_code == 422
    assert s.active_version("credit_analysis") == v["id"]


def test_remote_schema_reference_rejected(setup):
    c, _s, _v, _a = setup
    result = c.post(
        "/api/versions",
        json={
            "task": "credit_analysis",
            "prompt": "p",
            "schema": {"$ref": "https://example.invalid/schema"},
            "name": "bad",
        },
    )
    assert result.status_code == 422


def test_running_job_stop_uses_process_group(tmp_path, monkeypatch):
    from unittest.mock import Mock

    s = Store(tmp_path)
    process = FakeProcess()
    process.wait = Mock()
    process.pid = 123456
    jobs = Jobs(s, lambda *args, **kwargs: process)
    record = jobs.enqueue({"kind": "train"})
    jobs.tick()
    signals = []
    monkeypatch.setattr(
        "credit_risk.workbench.jobs.os.killpg", lambda pid, sig: signals.append((pid, sig))
    )
    jobs.stop(record["id"])
    process.code = -15
    jobs.tick()
    assert signals[0][0] == 123456
    assert s.get("job", record["id"])["status"] == "cancelled"


def test_job_detail_reports_progress_and_downloads_full_log(tmp_path, monkeypatch):
    import yaml

    from credit_risk.workbench import server

    monkeypatch.setattr(server, "PROJECT", tmp_path)
    workspace = tmp_path / "workspace"
    app = create_app(workspace, False)
    client = TestClient(app)
    store = app.state.store
    identity = "progress-job"
    output = workspace / "runs" / identity
    output.mkdir(parents=True)
    (output / "training.yaml").write_text(yaml.safe_dump({"iters": 128}))
    (output / "job.log").write_text("complete local log\n")
    adapter = tmp_path / "adapters/candidates/credit_analysis" / identity
    adapter.mkdir(parents=True)
    (adapter / "metrics.jsonl").write_text(
        json.dumps({"train": {"iteration": 8, "train_loss": 1.0}}) + "\n"
    )
    store.add(
        "job",
        {
            "status": "running",
            "spec": {
                "kind": "train",
                "task": "credit_analysis",
                "output": str(output),
                "config": {"grad_accumulation_steps": 8},
            },
        },
        identity,
    )
    detail = client.get(f"/api/jobs/{identity}").json()
    assert detail["progress"] == {
        "current_micro_batches": 8,
        "total_micro_batches": 128,
        "current_optimizer_updates": 1,
        "total_optimizer_updates": 16,
        "percent": 6.2,
    }
    download = client.get(f"/api/jobs/{identity}/log")
    assert download.text == "complete local log\n"
    assert f'filename="{identity}.log"' in download.headers["content-disposition"]


def test_one_click_comparison_queues_matching_base_and_candidate(tmp_path, monkeypatch):
    from credit_risk.workbench import server
    from credit_risk.workbench.evaluation import EVALUATOR_VERSION

    monkeypatch.setattr(server, "PROJECT", tmp_path)
    model = {"id": "base-revision", "path": str(tmp_path / "model"), "label": "Fixture base"}
    monkeypatch.setattr(server, "model_catalog", lambda: [model])
    app = create_app(tmp_path / "workspace", False)
    client = TestClient(app)
    client.headers["X-Workbench-Token"] = client.get("/api/session").json()["token"]
    store = app.state.store
    version = store.get("version", store.active_version("credit_analysis"))
    cases = [
        case(case_id="v", group_id="vg", split="validation"),
        case(case_id="t", group_id="tg", split="test", question="test question"),
        case(
            case_id="o",
            group_id="og",
            split="oot",
            question="oot question",
            as_of_date="2026-01-01",
        ),
    ]
    dataset = inspect_dataset(manifest(tmp_path / "dataset", cases))
    dataset = store.add("dataset", dataset, dataset["hash"])
    training_id = "completed-training"
    output = tmp_path / "workspace/runs" / training_id
    output.mkdir(parents=True)
    adapter = tmp_path / "adapters/candidates/credit_analysis" / training_id
    adapter.mkdir(parents=True)
    best = b"verified-best"
    final = b"verified-final"
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "best_adapters.safetensors").write_bytes(best)
    (adapter / "adapters.safetensors").write_bytes(final)
    (adapter / "completion.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "best_optimizer_updates": 1,
                "checkpoint_sha256": hashlib.sha256(best).hexdigest(),
                "final_checkpoint_sha256": hashlib.sha256(final).hexdigest(),
            }
        )
    )
    training = store.add(
        "job",
        {
            "status": "completed",
            "spec": {
                "kind": "train",
                "task": "credit_analysis",
                "dataset": dataset,
                "model": model,
                "version": version,
                "config": Config().model_dump(),
                "output": str(output),
            },
        },
        training_id,
    )
    request = {
        "training_job_id": training["id"],
        "checkpoint": "best",
        "splits": ["validation", "test", "oot"],
        "generation_profile": "deterministic",
    }
    created = client.post("/api/comparison-runs", json=request)
    assert created.status_code == 200, created.text
    payload = created.json()
    base = store.get("job", payload["base_evaluation_job_id"])
    candidate = store.get("job", payload["candidate_evaluation_job_id"])
    assert "adapter_path" not in base["spec"]
    assert candidate["spec"]["adapter_job_id"] == training_id
    assert candidate["spec"]["checkpoint_sha256"] == hashlib.sha256(best).hexdigest()
    for key in (
        "dataset_manifest",
        "case_set_hash",
        "model_revision",
        "prompt_hash",
        "schema_hash",
        "generation_hash",
        "evaluator_version",
    ):
        assert base["spec"]["comparison_compatibility"][key] == candidate["spec"][
            "comparison_compatibility"
        ][key]
    assert base["spec"]["comparison_compatibility"]["evaluator_version"] == EVALUATOR_VERSION
    assert client.post("/api/comparison-runs", json=request).status_code == 422
    queued = client.get("/api/comparison-runs/" + payload["comparison_id"]).json()
    assert queued["status"] == "queued"
    assert queued["base"]["progress"] == {"current_cases": 0, "total_cases": 3, "percent": 0.0}


def test_completed_pair_returns_automatic_comparison(tmp_path, monkeypatch):
    from credit_risk.workbench import server

    monkeypatch.setattr(server, "PROJECT", tmp_path)
    model = {"id": "base-revision", "path": str(tmp_path / "model"), "label": "Fixture base"}
    monkeypatch.setattr(server, "model_catalog", lambda: [model])
    app = create_app(tmp_path / "workspace", False)
    client = TestClient(app)
    client.headers["X-Workbench-Token"] = client.get("/api/session").json()["token"]
    store = app.state.store
    version = store.get("version", store.active_version("credit_analysis"))
    cases = [case(case_id="v", group_id="vg", split="validation")]
    dataset = inspect_dataset(manifest(tmp_path / "dataset", cases))
    dataset = store.add("dataset", dataset, dataset["hash"])
    training_id = "training-for-comparison"
    output = tmp_path / "workspace/runs" / training_id
    output.mkdir(parents=True)
    adapter = tmp_path / "adapters/candidates/credit_analysis" / training_id
    adapter.mkdir(parents=True)
    weights = b"best"
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "best_adapters.safetensors").write_bytes(weights)
    (adapter / "adapters.safetensors").write_bytes(b"final")
    (adapter / "completion.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "best_optimizer_updates": 1,
                "checkpoint_sha256": hashlib.sha256(weights).hexdigest(),
                "final_checkpoint_sha256": hashlib.sha256(b"final").hexdigest(),
            }
        )
    )
    store.add(
        "job",
        {
            "status": "completed",
            "spec": {
                "kind": "train",
                "task": "credit_analysis",
                "dataset": dataset,
                "model": model,
                "version": version,
                "config": Config().model_dump(),
                "output": str(output),
            },
        },
        training_id,
    )
    pair = client.post(
        "/api/comparison-runs",
        json={"training_job_id": training_id, "splits": ["validation"]},
    ).json()
    report = {
        "evaluator_version": "field-checks-v2",
        "case_set_hash": store.get("comparison", pair["comparison_id"])["compatibility"][
            "case_set_hash"
        ],
        "identity": {"version_id": version["id"], "generation": {"profile": "same"}},
        "splits": {
            "validation": {
                "json_validity": {
                    "value": 1.0,
                    "denominator": 1,
                    "ci95": [1.0, 1.0],
                    "sufficient_sample": False,
                }
            }
        },
        "cases": [],
    }
    for key in ("base_evaluation_job_id", "candidate_evaluation_job_id"):
        job_id = pair[key]
        run = store.get("job", job_id)
        Path(run["spec"]["output"]).joinpath("result.json").write_text(json.dumps(report))
        store.update_job(job_id, status="completed")
    result = client.get("/api/comparison-runs/" + pair["comparison_id"])
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "completed"
    card = result.json()["metrics"]["scorecards"]["validation"]["json_validity"]
    assert card == {"base": 1.0, "candidate": 1.0, "delta": 0.0, "denominator": 1}


def test_fuse_selects_best_checkpoint_and_records_hash(tmp_path, monkeypatch):
    import sys

    import yaml

    from credit_risk import training

    adapter = tmp_path / "adapters"
    adapter.mkdir()
    model = tmp_path / "base"
    model.mkdir()
    (adapter / "completion.json").write_text(json.dumps({
        "status": "completed", "manifest_version": 2, "best_optimizer_updates": 1,
        "baseline_loss": 1.0, "selected_loss": .8,
        "checkpoint_sha256": hashlib.sha256(b"best-unit-fixture").hexdigest(),
    }))
    (adapter / "adapter_config.json").write_text(json.dumps({"model": str(model)}))
    (adapter / "best_adapters.safetensors").write_bytes(b"best-unit-fixture")
    (adapter / "adapters.safetensors").write_bytes(b"final-unit-fixture")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model": "unresolved-id", "adapter_path": str(adapter)}))
    dest = tmp_path / "models/candidates/export"
    monkeypatch.setattr(training, "DEFAULT_MODEL_DIR", tmp_path / "models/candidates")

    def fuse(command, check):
        from pathlib import Path

        chosen = Path(command[command.index("--adapter-path") + 1])
        assert (chosen / "adapters.safetensors").read_bytes() == b"best-unit-fixture"
        assert command[command.index("--model") + 1] == str(model)
        dest.mkdir(parents=True)

    monkeypatch.setattr(training.subprocess, "run", fuse)
    monkeypatch.setattr(
        sys,
        "argv",
        ["training", "fuse", "--config", str(config), "--save-path", str(dest), "--execute"],
    )
    training.main()
    assert (
        json.loads((dest / "export_manifest.json").read_text())["checkpoint_sha256"]
        == hashlib.sha256(b"best-unit-fixture").hexdigest()
    )


def test_preflight_empty_phase_two_data_blocks_start(setup):
    c, s, v, _a = setup
    from credit_risk.workbench.worker import model_catalog

    models = model_catalog()
    if not models:
        pytest.skip("No cached base; normal empty state")
    request = {
        "task": "credit_analysis",
        "kind": "train",
        "version_id": v["id"],
        "model_id": models[0]["id"],
    }
    result = c.post("/api/jobs", json=request)
    assert result.status_code == 422
    assert s.list("job") == []


def test_invalid_outputs_stay_in_expected_metric_denominators():
    v = default_version("credit_analysis")
    v["schema"]["properties"]["value"] = {"type": "number"}
    a = case(case_id="a", expected={"numerics": {"value": 2}, "must_abstain": True})
    b = case(case_id="b", expected={"numerics": {"value": 2}, "must_abstain": True})
    result = evaluate(
        [a, b], lambda c, v: {**answer(), "value": 2} if c.case_id == "a" else "malformed", v, {}
    )
    card = result["splits"]["development"]
    assert card["numerical_agreement"]["value"] == 0.5
    assert card["numerical_agreement"]["denominator"] == 2
    assert card["numerical_agreement"]["ci95"] == [0.0, 1.0]
    assert card["abstention_recall"]["value"] == 0.5
    assert card["abstention_recall"]["denominator"] == 2


def test_v2_manifest_requires_versioned_lineage(tmp_path):
    item = case(
        split="train",
        target=answer(),
        provenance={
            "classification": "synthetic",
            "source_snapshot_hash": "a" * 64,
            "template_family": "manual-credit-review",
            "transformations": ["masked-identifiers"],
        },
    )
    path = manifest(tmp_path, [item])
    payload = json.loads(path.read_text())
    payload.update(
        format="credit-workbench-v2",
        dataset_id="synthetic-credit",
        dataset_version="2026-09-09.1",
        created_at="2026-09-09T00:00:00Z",
    )
    path.write_text(json.dumps(payload))
    inspected = inspect_dataset(path)
    assert inspected["contract_version"] == "credit-workbench-v2"
    assert inspected["contract_warnings"] == []
    del payload["dataset_version"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="dataset_version"):
        inspect_dataset(path)


def test_v2_masked_data_requires_versioned_privacy_scan(tmp_path):
    item = case(
        split="train",
        target=answer(),
        provenance={
            "classification": "masked",
            "source_snapshot_hash": "b" * 64,
            "template_family": "masked-credit",
            "transformations": ["tokenized-identifiers"],
        },
    )
    path = manifest(tmp_path, [item])
    payload = json.loads(path.read_text())
    payload.update(
        format="credit-workbench-v2",
        dataset_id="masked-credit",
        dataset_version="1",
        created_at="2026-09-09T00:00:00Z",
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="privacy scan"):
        inspect_dataset(path)
    payload["privacy_scan"] = {
        "status": "passed",
        "scanner_version": "fixture-scanner-v1",
    }
    path.write_text(json.dumps(payload))
    assert inspect_dataset(path)["contract_version"] == "credit-workbench-v2"


def test_numeric_contract_supports_tolerance_unit_and_date():
    output = {
        **answer("ANSWERED"),
        "risk_driver_details": [
            {
                "driver": "days_past_due",
                "observed_value": 47.1,
                "unit": "days",
                "direction": "deteriorating",
                "severity": "high",
                "as_of_date": "2025-01-01",
                "evidence_ids": [],
            }
        ],
    }
    expected = {
        "numerics": {
            "risk_driver_details.0.observed_value": {
                "value": 47,
                "abs_tolerance": 0.2,
                "unit": "days",
                "as_of_date": "2025-01-01",
            }
        }
    }
    report = evaluate(
        [case(expected=expected)], lambda _case, _version: output, default_version("credit_analysis"), {}
    )
    assert report["splits"]["development"]["numerical_agreement"]["value"] == 1
    output["risk_driver_details"][0]["unit"] = "months"
    report = evaluate(
        [case(expected=expected)], lambda _case, _version: output, default_version("credit_analysis"), {}
    )
    assert report["splits"]["development"]["numerical_agreement"]["value"] == 0


def test_training_parameter_surface_is_bounded():
    configured = Config(
        optimizer="adamw",
        weight_decay=0.01,
        schedule="cosine_decay",
        warmup_ratio=0.05,
        num_layers=32,
        target_modules="attention_mlp",
        lora_parameters={"rank": 32, "scale": 2, "dropout": 0},
    )
    assert configured.optimizer == "adamw"
    assert configured.lora_parameters.rank == 32
    with pytest.raises(ValueError):
        Config(lora_parameters={"rank": 64, "scale": 2, "dropout": 0.05})


def test_workbench_training_requires_v2_contract(tmp_path):
    from credit_risk.workbench.worker import prepare_training

    train = case(split="train", target=answer())
    validation = case(
        case_id="validation",
        group_id="validation-group",
        split="validation",
        question="validation question",
        target=answer(),
    )
    path = manifest(tmp_path / "legacy", [train, validation])
    dataset = inspect_dataset(path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    spec = {
        "task": "credit_analysis",
        "dataset": dataset,
        "version": {**default_version("credit_analysis"), "id": "version"},
        "model": {"path": str(model)},
        "config": Config().model_dump(),
        "output": str(tmp_path / "candidate"),
    }
    with pytest.raises(ValueError, match="v2 data contract"):
        prepare_training(spec, tokenizer=object())


def test_v2_preflight_resolves_schedule_modules_and_template_mode(tmp_path):
    pytest.importorskip("mlx_lm")
    from credit_risk.workbench.worker import MLP_KEYS, prepare_training

    def v2_case(split, identity):
        return case(
            case_id=identity,
            group_id=identity,
            split=split,
            question=identity,
            target=answer(),
            provenance={
                "classification": "synthetic",
                "source_snapshot_hash": ("a" if split == "train" else "b") * 64,
                "template_family": identity,
                "transformations": [],
            },
        )

    path = manifest(
        tmp_path / "v2", [v2_case("train", "train"), v2_case("validation", "validation")]
    )
    payload = json.loads(path.read_text())
    payload.update(
        format="credit-workbench-v2",
        dataset_id="schedule-test",
        dataset_version="1",
        created_at="2026-09-09T00:00:00Z",
    )
    path.write_text(json.dumps(payload))
    dataset = inspect_dataset(path)
    model = tmp_path / "model-v2"
    model.mkdir()
    quantization = {"bits": 4, "group_size": 64, "mode": "affine"}
    (model / "config.json").write_text(json.dumps({"quantization": quantization}))
    calls = []

    class Tokenizer:
        chat_template = "fixture-template"

        def apply_chat_template(self, turns, add_generation_prompt=False, **kwargs):
            calls.append(kwargs.get("enable_thinking"))
            return list(range(4 + 2 * len(turns) + int(add_generation_prompt)))

    config = Config(
        optimizer="adamw",
        weight_decay=0.01,
        schedule="cosine_decay",
        warmup_ratio=0.05,
        target_modules="attention_mlp",
    ).model_dump()
    cfg, metadata = prepare_training(
        {
            "task": "credit_analysis",
            "dataset": dataset,
            "version": {**default_version("credit_analysis"), "id": "v"},
            "model": {"path": str(model)},
            "config": config,
            "output": str(tmp_path / "candidate-v2"),
        },
        tokenizer=Tokenizer(),
    )
    assert calls and set(calls) == {False}
    assert cfg["optimizer_config"]["adamw"]["weight_decay"] == 0.01
    assert cfg["lr_schedule"]["name"] == "cosine_decay"
    assert set(MLP_KEYS).issubset(cfg["lora_parameters"]["keys"])
    assert metadata["chat_template_mode"] == {"enable_thinking": False}
    assert metadata["quantization"] == quantization


def test_unknown_correction_stays_untriaged_and_out_of_training(setup):
    _client, store, _version, captured = setup
    result = submit(store, captured["id"], "unknown-cause", "wrong", answer())
    assert result["correction_valid"]
    assert result["eligibility_status"] == "untriaged"
    assert not result["eligible_for_training"]
    assert batch_records(store, "credit_analysis")["cases"] == []


def test_correction_cannot_create_its_own_expected_result(setup):
    _client, store, _version, captured = setup
    captured = store.add(
        "answer",
        {
            **{k: v for k, v in captured.items() if k not in ("id", "created_at")},
            "case": case(expected={}).model_dump(),
        },
    )
    result = submit(
        store,
        captured["id"],
        "no-independent-label",
        "correct this",
        answer(),
        cause="model_behaviour",
    )
    assert result["correction_valid"]
    assert result["eligibility_status"] == "needs_expected_results"
    assert not result["has_independent_expectations"]
    assert not result["eligible_for_training"]
    assert store.list("regression") == []


def test_feedback_batch_deduplicates_equivalent_corrections(setup):
    _client, store, _version, captured = setup
    second = store.add(
        "answer", {k: v for k, v in captured.items() if k not in ("id", "created_at")}
    )
    submit(
        store,
        captured["id"],
        "duplicate-a",
        "wrong",
        answer(),
        cause="model_behaviour",
    )
    submit(
        store,
        second["id"],
        "duplicate-b",
        "same correction",
        answer(),
        cause="model_behaviour",
    )
    batch = batch_records(store, "credit_analysis")
    assert len(batch["cases"]) == 1
    assert len(batch["duplicate_or_capped"]) == 1


def test_confidence_calibration_is_reported_but_small_samples_are_marked():
    output = {
        **answer("ANSWERED"),
        "inferences": [{"statement": "", "basis": "", "confidence": 0.8}],
    }
    report = evaluate(
        [case(expected={"confidence_labels": {"inferences.0.confidence": 1}})],
        lambda _case, _version: output,
        default_version("credit_analysis"),
        {},
    )
    metric = report["splits"]["development"]["confidence_brier_score"]
    assert metric["value"] == pytest.approx(0.04)
    calibration = report["calibration"]["development"]
    assert calibration["brier_score"] == pytest.approx(0.04)
    assert calibration["expected_calibration_error"] == pytest.approx(0.2)
    assert calibration["status"] == "Small sample"


def test_paired_scorecard_keeps_metrics_with_unequal_or_one_sided_denominators():
    from credit_risk.workbench.evaluation import EVALUATOR_VERSION, aggregate

    def report(rows):
        return {
            "case_set_hash": "same",
            "evaluator_version": EVALUATOR_VERSION,
            "identity": {"generation": {"profile": "same"}, "version_id": "v"},
            "splits": {"test": aggregate(rows)},
            "portfolios": {"retail": aggregate(rows)},
            "cases": rows,
        }

    base = [
        {"case_id": "a", "split": "test", "metrics": {"json_validity": 0.0}, "failures": ["x"]},
        {
            "case_id": "b",
            "split": "test",
            "metrics": {"json_validity": 1.0, "citation_resolution": 0.5},
            "failures": [],
        },
    ]
    candidate = [
        {
            "case_id": case_id,
            "split": "test",
            "metrics": {
                "json_validity": 1.0,
                "citation_resolution": 1.0,
                "confidence_brier_score": 0.1,
            },
            "failures": [],
        }
        for case_id in ("a", "b")
    ]
    result = compare(report(base), report(candidate))
    card = result["scorecards"]["test"]
    assert card["json_validity"] == {"base": 0.5, "candidate": 1.0, "delta": 0.5, "denominator": 2}
    assert card["citation_resolution"]["delta"] == 0.5
    assert card["citation_resolution"]["base_denominator"] == 1
    assert card["citation_resolution"]["candidate_denominator"] == 2
    assert card["confidence_brier_score"]["base"] is None
    assert card["confidence_brier_score"]["lower_is_better"] is True
    assert result["portfolios"]["retail"]["citation_resolution"]["candidate"] == 1.0


def test_ended_comparison_side_cancels_queued_partner(tmp_path):
    s = Store(tmp_path)
    launched = []

    def launch(*args, **kw):
        launched.append(FakeProcess())
        return launched[-1]

    jobs = Jobs(s, launch)
    base = jobs.enqueue({"kind": "evaluate", "comparison_id": "pair", "comparison_role": "base"})
    candidate = jobs.enqueue(
        {"kind": "evaluate", "comparison_id": "pair", "comparison_role": "candidate"}
    )
    unrelated = jobs.enqueue({"kind": "evaluate"})
    jobs.tick()
    launched[0].code = 1
    jobs.tick()
    assert s.get("job", base["id"])["status"] == "failed"
    assert s.get("job", candidate["id"])["status"] == "cancelled"
    assert "base job ended as failed" in s.get("job", candidate["id"])["error"]
    assert s.get("job", unrelated["id"])["status"] == "running"

    first = jobs.enqueue({"kind": "evaluate", "comparison_id": "other", "comparison_role": "base"})
    second = jobs.enqueue(
        {"kind": "evaluate", "comparison_id": "other", "comparison_role": "candidate"}
    )
    jobs.stop(first["id"])
    assert s.get("job", second["id"])["status"] == "cancelled"


def test_restart_never_starts_previously_queued_jobs(tmp_path):
    s = Store(tmp_path)
    launched = []
    queued = Jobs(s).enqueue({"kind": "evaluate"})
    jobs = Jobs(s, lambda *a, **k: launched.append(FakeProcess()) or launched[-1])
    jobs.start()
    jobs.close()
    jobs.tick()
    assert launched == []
    assert s.get("job", queued["id"])["status"] == "interrupted"


def test_recommended_contract_numbering_continues_and_is_reported(tmp_path):
    store = Store(tmp_path)
    legacy = default_version("credit_analysis")
    legacy["prompt"] = "Older prompt"
    legacy["name"] = "Recommended structured contract v3"
    store.active_version("credit_analysis", store.add("version", legacy)["id"])
    client = TestClient(create_app(tmp_path, False))
    state = client.get("/api/state").json()
    names = [v["name"] for v in state["versions"] if v["task"] == "credit_analysis"]
    assert names.count("Recommended structured contract v3") == 1
    recommended = next(v for v in state["versions"] if v["id"] == state["recommended"]["credit_analysis"])
    assert recommended["name"] == "Recommended structured contract v4"
    assert state["active"]["credit_analysis"] != recommended["id"]
    assert state["evaluation_policy"]["min_reportable_slice"] == 10


def test_state_omits_frozen_cases_and_intervals_follow_accumulation(tmp_path, monkeypatch):
    from credit_risk.workbench import server

    model = {"id": "base-revision", "path": str(tmp_path / "model"), "label": "Fixture base"}
    monkeypatch.setattr(server, "model_catalog", lambda: [model])
    app = create_app(tmp_path / "workspace", False)
    client = TestClient(app)
    client.headers["X-Workbench-Token"] = client.get("/api/session").json()["token"]
    store = app.state.store
    dataset = inspect_dataset(
        manifest(tmp_path / "dataset", [case(case_id="t", group_id="tg", split="test")])
    )
    dataset = store.add("dataset", dataset, dataset["hash"])
    request = {
        "task": "credit_analysis",
        "kind": "evaluate",
        "dataset_id": dataset["id"],
        "version_id": store.active_version("credit_analysis"),
        "model_id": model["id"],
        "splits": ["test"],
    }
    expected = {16: (16, 80, 80), 32: (32, 96, 96), 8: (8, 80, 80)}
    for accumulation, intervals in expected.items():
        job = client.post(
            "/api/jobs", json={**request, "config": {"grad_accumulation_steps": accumulation}}
        )
        assert job.status_code == 200, job.text
        config = job.json()["spec"]["config"]
        assert (config["steps_per_report"], config["steps_per_eval"], config["save_every"]) == intervals
    explicit = {"grad_accumulation_steps": 16, "steps_per_report": 8}
    assert client.post("/api/jobs", json={**request, "config": explicit}).status_code == 422
    listed = client.get("/api/state").json()["jobs"][0]["spec"]["dataset"]
    assert "cases" not in listed and listed["counts"]["test"] == 1
    assert store.list("job")[0]["spec"]["dataset"]["cases"]


def test_preflight_token_lengths_are_recorded_for_the_dataset(tmp_path, monkeypatch):
    from credit_risk.workbench import server

    model = {"id": "base-revision", "path": str(tmp_path / "model"), "label": "Fixture base"}
    monkeypatch.setattr(server, "model_catalog", lambda: [model])
    metadata = {
        "token_lengths": [100, 300, 200],
        "max_tokens": 300,
        "min_assistant_tokens": 20,
        "chat_template_hash": "template",
    }
    monkeypatch.setattr(server, "prepare_training", lambda spec: ({}, metadata))
    app = create_app(tmp_path / "workspace", False)
    client = TestClient(app)
    client.headers["X-Workbench-Token"] = client.get("/api/session").json()["token"]
    store = app.state.store
    dataset = inspect_dataset(manifest(tmp_path / "dataset", [case(case_id="t", split="train")]))
    dataset = store.add("dataset", dataset, dataset["hash"])
    request = {
        "task": "credit_analysis",
        "kind": "train",
        "dataset_id": dataset["id"],
        "version_id": store.active_version("credit_analysis"),
        "model_id": model["id"],
    }
    for _ in range(2):
        assert client.post("/api/preflight", json=request).status_code == 200
    [record] = client.get("/api/state").json()["preflights"]
    assert record["dataset_id"] == dataset["id"]
    assert (record["examples"], record["max_tokens"], record["p95_tokens"]) == (3, 300, 300)


def test_feedback_fragment_explains_skipped_feedback(setup):
    _c, s, _v, a = setup
    frozen = s.add(
        "answer",
        {
            **{k: v for k, v in a.items() if k not in ("id", "created_at")},
            "case": case(split="test").model_dump(),
        },
    )
    submit(s, frozen["id"], "frozen", "wrong", answer(), "model_behaviour")
    submit(s, a["id"], "comment-only", "looks off")
    batch = batch_records(s, "credit_analysis")
    assert batch["cases"] == []
    assert batch["skipped_reasons"] == {"protected_split_or_group": 1, "submitted": 1}


def test_jobs_record_start_and_finish_times(tmp_path):
    s = Store(tmp_path)
    launched = []
    jobs = Jobs(s, lambda *a, **k: launched.append(FakeProcess()) or launched[-1])
    job = jobs.enqueue({"kind": "evaluate"})
    jobs.tick()
    running = s.get("job", job["id"])
    assert running["started_at"] and "finished_at" not in running
    launched[0].code = 0
    jobs.tick()
    finished = s.get("job", job["id"])
    assert finished["finished_at"] >= finished["started_at"]


def _timing_app(tmp_path, monkeypatch):
    from credit_risk.workbench import server

    monkeypatch.setattr(server, "PROJECT", tmp_path)
    app = create_app(tmp_path / "workspace", False)
    return app, TestClient(app)


def _ago(seconds):
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()


def test_evaluation_timing_excludes_model_load_and_estimates_remaining(tmp_path, monkeypatch):
    app, client = _timing_app(tmp_path, monkeypatch)
    output = tmp_path / "workspace/runs/eval"
    output.mkdir(parents=True)
    (output / "progress.json").write_text(
        json.dumps(
            {"current_cases": 4, "total_cases": 10, "started_at": _ago(90), "updated_at": _ago(10)}
        )
    )
    spec = {"kind": "evaluate", "task": "credit_analysis", "output": str(output), "splits": ["test"]}
    app.state.store.add(
        "job", {"status": "running", "started_at": _ago(300), "spec": spec}, "eval"
    )
    timing = client.get("/api/jobs/eval").json()["timing"]
    assert timing["unit"] == "case"
    assert timing["seconds_per_unit"] == pytest.approx(20, abs=0.5)
    assert timing["eta_seconds"] == pytest.approx(110, abs=3)
    assert timing["elapsed_seconds"] == pytest.approx(300, abs=3)


def test_training_timing_uses_reported_metrics(tmp_path, monkeypatch):
    import yaml

    app, client = _timing_app(tmp_path, monkeypatch)
    output = tmp_path / "workspace/runs/train"
    output.mkdir(parents=True)
    (output / "training.yaml").write_text(yaml.safe_dump({"iters": 40}))
    adapter = tmp_path / "adapters/candidates/credit_analysis/train"
    adapter.mkdir(parents=True)
    (adapter / "metrics.jsonl").write_text(
        json.dumps({"validation": {"iteration": 0, "val_loss": 1.0, "reported_at": _ago(100)}})
        + "\n"
        + json.dumps({"train": {"iteration": 10, "train_loss": 0.9, "reported_at": _ago(0)}})
        + "\n"
    )
    spec = {
        "kind": "train",
        "task": "credit_analysis",
        "output": str(output),
        "config": {"grad_accumulation_steps": 2},
    }
    app.state.store.add("job", {"status": "running", "started_at": _ago(400), "spec": spec}, "train")
    timing = client.get("/api/jobs/train").json()["timing"]
    assert timing["unit"] == "micro-batch"
    assert timing["seconds_per_unit"] == pytest.approx(10, abs=0.5)
    assert timing["eta_seconds"] == pytest.approx(300, abs=5)


def test_comparison_estimates_queued_candidate_from_base_rate(tmp_path, monkeypatch):
    app, client = _timing_app(tmp_path, monkeypatch)
    store = app.state.store
    ids = {}
    for role in ("base", "candidate"):
        output = tmp_path / "workspace/runs" / role
        output.mkdir(parents=True)
        spec = {
            "kind": "evaluate",
            "task": "credit_analysis",
            "output": str(output),
            "splits": ["test"],
            "cases": [{"split": "test"}] * 10,
            "comparison_role": role,
        }
        ids[role] = store.add("job", {"status": "queued", "spec": spec}, role)["id"]
    (tmp_path / "workspace/runs/base/progress.json").write_text(
        json.dumps(
            {"current_cases": 5, "total_cases": 10, "started_at": _ago(50), "updated_at": _ago(0)}
        )
    )
    store.update_job("base", status="running", started_at=_ago(60))
    store.add(
        "comparison",
        {"base_job_id": ids["base"], "candidate_job_id": ids["candidate"], "checkpoint": "best"},
        "pair",
    )
    pair = client.get("/api/comparison-runs/pair").json()
    assert pair["status"] == "running"
    assert pair["timing"]["eta_seconds"] == pytest.approx(50 + 100, abs=3)
    listed = client.get("/api/state").json()["jobs"]
    assert next(j for j in listed if j["id"] == "base")["started_at"]
    assert next(j for j in listed if j["id"] == "candidate")["started_at"] is None


def test_comparison_against_previous_adapter_uses_reference_checkpoint(tmp_path, monkeypatch):
    from credit_risk.workbench import server

    monkeypatch.setattr(server, "PROJECT", tmp_path)
    model = {"id": "base-revision", "path": str(tmp_path / "model"), "label": "Fixture base"}
    monkeypatch.setattr(server, "model_catalog", lambda: [model])
    app = create_app(tmp_path / "workspace", False)
    client = TestClient(app)
    client.headers["X-Workbench-Token"] = client.get("/api/session").json()["token"]
    store = app.state.store
    version = store.get("version", store.active_version("credit_analysis"))
    dataset = inspect_dataset(manifest(tmp_path / "dataset", [case(case_id="v", group_id="vg", split="validation")]))
    dataset = store.add("dataset", dataset, dataset["hash"])

    def trained(identity, weights):
        output = tmp_path / "workspace/runs" / identity
        output.mkdir(parents=True)
        adapter = tmp_path / "adapters/candidates/credit_analysis" / identity
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "best_adapters.safetensors").write_bytes(weights)
        (adapter / "adapters.safetensors").write_bytes(weights)
        digest_value = hashlib.sha256(weights).hexdigest()
        (adapter / "completion.json").write_text(json.dumps({"status": "completed", "best_optimizer_updates": 1, "checkpoint_sha256": digest_value, "final_checkpoint_sha256": digest_value}))
        spec = {"kind": "train", "task": "credit_analysis", "dataset": dataset, "model": model, "version": version, "config": Config().model_dump(), "output": str(output)}
        return store.add("job", {"status": "completed", "spec": spec}, identity), digest_value

    previous, previous_hash = trained("previous-adapter", b"old")
    candidate, candidate_hash = trained("new-adapter", b"new")
    request = {"training_job_id": candidate["id"], "reference_job_id": previous["id"], "splits": ["validation"]}
    pair = client.post("/api/comparison-runs", json=request)
    assert pair.status_code == 200, pair.text
    base = store.get("job", pair.json()["base_evaluation_job_id"])
    other = store.get("job", pair.json()["candidate_evaluation_job_id"])
    assert base["spec"]["adapter_job_id"] == previous["id"] and base["spec"]["checkpoint_sha256"] == previous_hash
    assert other["spec"]["adapter_job_id"] == candidate["id"] and other["spec"]["checkpoint_sha256"] == candidate_hash
    assert store.get("comparison", pair.json()["comparison_id"])["reference_job_id"] == previous["id"]
    same = client.post("/api/comparison-runs", json={**request, "reference_job_id": candidate["id"]})
    assert same.status_code == 422
