"""Phase 4: verified replay, consistency gates and building a dataset version from feedback."""

import importlib.util
import json
from pathlib import Path

from test_ask import Model, ask, confirm, get, serve  # noqa: F401 — shared Ask fixtures
from test_ask import app  # noqa: F401 — pytest fixture reused by name

from credit_risk.workbench import worker
from credit_risk.workbench.learning import consistency_gates, load_gates
from credit_risk.workbench.store import Store

ROOT = Path(__file__).resolve().parents[2]


def plan(as_of="2026-01-15", date_to="2025-12-31"):
    return {
        "portfolio": "retail",
        "jurisdiction": "SAMA",
        "obligor_id": "OBL-0002",
        "date_from": "2025-01-01",
        "date_to": date_to,
        "as_of_date": as_of,
        "metrics": ["stage", "pit_pd"],
        "analysis_type": "factsheet",
    }


def verified_answer(app, as_of="2026-01-15", date_to="2025-12-31"):  # noqa: F811
    import test_ask

    original = test_ask.AS_OF
    test_ask.AS_OF = as_of
    try:
        question = ask(app, "What is the current credit stage of OBL-0002?", draft=False)
    finally:
        test_ask.AS_OF = original
    confirm(app, question, plan(as_of, date_to))
    serve(app)
    answer = get(app, question)["answer"]
    assert app.state.client.post(f"/api/answers/{answer['id']}/verify", json={}).status_code == 200
    return answer


def test_replay_matches_verified_answers_and_lists_drift(app):  # noqa: F811
    answer = verified_answer(app)
    client = app.state.client
    assert client.get("/api/verified").json()[0]["answer_id"] == answer["id"]
    job = client.post("/api/replay", json={"model": {"model_id": "base-revision"}}).json()
    assert job["spec"]["kind"] == "replay" and job["spec"]["answers"] == [answer["id"]]
    worker.run_replay(job["spec"], provider_factory=lambda spec: app.state.fake)
    result = json.loads((Path(job["spec"]["output"]) / "result.json").read_text())
    assert result["passed"] and result["matched"] == 1

    original = app.state.fake.__call__

    class Drifted(Model):
        def __call__(self, chat, seed):
            answer = json.loads(original(chat, seed))
            answer["risk_drivers"] = ["new driver"]
            return json.dumps(answer)

    drift = client.post("/api/replay", json={"model": {"model_id": "base-revision"}}).json()
    worker.run_replay(drift["spec"], provider_factory=lambda spec: Drifted())
    result = json.loads((Path(drift["spec"]["output"]) / "result.json").read_text())
    assert not result["passed"] and "risk_drivers" in result["mismatches"][0]["differences"]
    assert client.get("/api/jobs/" + drift["id"]).json()["timing"]["unit"] == "answer"


def test_consistency_gates_combine_metrics_replay_and_regressions(tmp_path):
    store = Store(tmp_path)
    gates = load_gates(ROOT / "configs/consistency_gates.yaml")

    def job(kind, result, **spec):
        output = tmp_path / "runs" / f"{kind}-{len(store.list('job'))}"
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps(result))
        return store.add("job", {"status": "completed", "spec": {"kind": kind, "task": "credit_analysis", "output": str(output), **spec}})

    def measured(value, n=12):
        return {"value": value, "denominator": n}

    candidate = {
        "job": {"spec": {"task": "credit_analysis"}},
        "result": {"splits": {
            "validation": {"repeated_agreement": measured(1.0), "equivalent_agreement": measured(1.0)},
            "test": {"repeated_agreement": measured(0.9)},
            "oot": {},
        }},
    }
    report = consistency_gates(store, gates, candidate, "train-1", "sha")
    by = {(r["gate"], r["scope"]): r["status"] for r in report["results"]}
    assert by[("repeated_agreement", "validation")] == "passed"
    assert by[("repeated_agreement", "test")] == "failed"
    assert by[("verified_replay", "verified answers")] == "not_evaluated"
    assert report["status"] == "failed"

    job("replay", {"total": 4, "matched": 4}, adapter_job_id="train-1", checkpoint_sha256="sha")
    job("regression", {"cases": [{"case_id": "a", "failures": []}, {"case_id": "b", "failures": []}]})
    job("regression", {"cases": [{"case_id": "a", "failures": []}, {"case_id": "b", "failures": ["x"]}]}, adapter_job_id="train-1")
    report = consistency_gates(store, gates, candidate, "train-1", "sha")
    by = {r["gate"]: r for r in report["results"]}
    assert by["verified_replay"]["status"] == "passed" and by["verified_replay"]["value"] == 1.0
    assert by["regression_checks"]["status"] == "failed" and by["regression_checks"]["newly_failing"] == ["b"]


def test_build_dataset_version_adds_feedback_with_paraphrase_family(app, tmp_path, monkeypatch):  # noqa: F811
    from credit_risk.workbench import server

    spec = importlib.util.spec_from_file_location("fixture_script", ROOT / "scripts/make_workbench_fixture.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    base = script.build(tmp_path / "curated.duckdb", tmp_path / "base-dataset", "base.1", {"train": 4, "validation": 2, "test": 2, "oot": 2})
    client = app.state.client
    registered = client.post("/api/datasets", json={"path": base["manifest"]})
    assert registered.status_code == 200, registered.text
    dataset = registered.json()

    answer = verified_answer(app, as_of="2025-12-31", date_to="2025-11-30")
    stage = answer["case"]["facts"]["current_position"]["stage"]
    statement = f"current_position.stage = {stage}"
    correction = {"answer_status": "ANSWERED", "executive_summary": statement, "facts": [{"statement": statement, "evidence_ids": [answer["case"]["case_id"]]}], "risk_drivers": []}
    feedback = client.post("/api/feedback", json={
        "submission_id": "fb-1", "interaction_id": answer["id"], "comment": "Drivers do not apply",
        "correction": correction, "cause": "model_behaviour", "expectations": {"fields": {"answer_status": "ANSWERED"}},
    }).json()
    assert feedback["eligibility_status"] == "eligible_for_batch", feedback

    monkeypatch.setattr(server, "PROJECT", tmp_path)
    built = client.post("/api/datasets/build", json={
        "base_dataset_id": dataset["id"], "dataset_version": "base.2", "feedback_fraction": 0.5,
        "paraphrases": {"fb-1": ["Which credit stage is OBL-0002 in right now?"]},
    })
    assert built.status_code == 200, built.text
    result = built.json()
    assert result["added_cases"] == 2 and result["paraphrase_cases"] == 1
    assert result["counts"] == {"train": 6, "validation": 2, "test": 2, "oot": 2}
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["derived_from"]["feedback_ids"] == ["fb-1"]
    train = [json.loads(line) for line in (Path(result["manifest_path"]).parent / "train.jsonl").read_text().splitlines()]
    family = [case for case in train if case.get("equivalence_id") == "feedback-fb-1"]
    assert len(family) == 2 and all(case["target"] == correction for case in family)
    for split in ("validation", "test", "oot"):
        assert (Path(result["manifest_path"]).parent / f"{split}.jsonl").read_bytes() == (Path(base["manifest"]).parent / f"{split}.jsonl").read_bytes()
    assert client.post("/api/datasets", json={"path": result["manifest_path"]}).status_code == 200
    again = client.post("/api/datasets/build", json={"base_dataset_id": dataset["id"], "dataset_version": "base.2"})
    assert again.status_code == 422
