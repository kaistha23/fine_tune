"""Local dashboard. Run: uv run python -m credit_risk.workbench.server"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from credit_risk.schemas import compact_json_schema
from credit_risk.workbench.contracts import Case, default_version, inspect_dataset
from credit_risk.review_store import digest
from credit_risk.workbench.evaluation import EVALUATOR_VERSION, check_schema, compare
from credit_risk.workbench.feedback import batch_records, submit
from credit_risk.workbench.jobs import Jobs
from credit_risk.workbench.store import Store
from credit_risk.workbench.worker import PROJECT, model_catalog, prepare_training


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Registration(Strict):
    path: str


class Version(Strict):
    task: Literal["credit_analysis", "query_plan"]
    prompt: str = Field(min_length=1, max_length=30000)
    schema_: dict = Field(alias="schema")
    name: str = Field(min_length=1, max_length=100)
    parent: str | None = None


class Feedback(Strict):
    semantic_review: dict | None = None
    submission_id: str = Field(min_length=1, max_length=128)
    interaction_id: str
    comment: str = Field(default="", max_length=8000)
    correction: dict | str | None = None
    cause: str = "unknown"
    expectations: dict | None = None


class LoRAParameters(Strict):
    rank: Literal[8, 16, 32] = 8
    scale: Literal[1.0, 2.0] = 2.0
    dropout: Literal[0.0, 0.05] = 0.0


class Config(Strict):
    batch_size: int = Field(default=1, ge=1, le=2)
    epochs: int = Field(default=2, ge=1, le=10)
    grad_accumulation_steps: int = Field(default=8, ge=1, le=32)
    max_seq_length: int = Field(default=1664, ge=256, le=8192)
    learning_rate: float = Field(default=2e-5, gt=0, le=0.001)
    num_layers: Literal[1, 4, 8, 16, 32] = 1
    target_modules: Literal["all_linear", "attention", "attention_mlp"] = "attention"
    optimizer: Literal["adam", "adamw"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0, le=0.1)
    schedule: Literal["constant", "cosine_decay"] = "constant"
    warmup_ratio: float = Field(default=0.0, ge=0, le=0.2)
    min_lr_ratio: float = Field(default=0.1, ge=0, le=1)
    seed: int = 42
    steps_per_report: int = Field(default=8, ge=1)
    steps_per_eval: int = Field(default=80, ge=1)
    save_every: int = Field(default=80, ge=1)
    val_batches: int = Field(default=-1, ge=-1)
    early_stopping_patience: int = Field(default=3, ge=1, le=10)
    early_stopping_min_delta: float = Field(default=0.001, ge=0, le=1)
    lora_parameters: LoRAParameters = Field(default_factory=LoRAParameters)


class JobRequest(Strict):
    task: Literal["credit_analysis", "query_plan"]
    kind: Literal["train", "evaluate", "regression"]
    dataset_id: str | None = None
    version_id: str
    model_id: str
    adapter_job_id: str | None = None
    checkpoint: Literal["best", "final"] = "best"
    generation_profile: Literal["deterministic", "serving"] = "deterministic"
    splits: list[Literal["train", "validation", "test", "oot", "development"]] = Field(
        default_factory=lambda: ["validation", "test", "oot"]
    )
    config: Config = Field(default_factory=Config)


class ComparisonRunRequest(Strict):
    training_job_id: str = Field(min_length=1, max_length=128)
    checkpoint: Literal["best", "final"] = "best"
    generation_profile: Literal["deterministic", "serving"] = "deterministic"
    splits: list[Literal["validation", "test", "oot"]] = Field(
        default_factory=lambda: ["validation", "test", "oot"]
    )


def create_app(root=None, start_scheduler=True):
    store = Store(root or PROJECT / "outputs/workbench")
    jobs = Jobs(store)
    token = secrets.token_urlsafe(32)
    for task in ("credit_analysis", "query_plan"):
        recommended = default_version(task)
        if store.active_version(task) is None:
            item = store.add("version", recommended)
            store.active_version(task, item["id"])
        elif not any(
            version["task"] == task
            and version["prompt"] == recommended["prompt"]
            and version["schema"] == recommended["schema"]
            for version in store.list("version")
        ):
            recommended.update(
                name="Recommended structured contract v2",
                parent=store.active_version(task),
            )
            store.add("version", recommended)

    @asynccontextmanager
    async def lifespan(app):
        if start_scheduler:
            jobs.start()
        yield
        if start_scheduler:
            jobs.close()

    app = FastAPI(
        title="Local Fine-tuning Workbench", lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.state.store = store
    app.state.jobs = jobs

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.url.hostname
        if host not in ("127.0.0.1", "localhost", "::1", "testserver"):
            return JSONResponse({"detail": "Localhost only"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)
        if request.headers.get("sec-fetch-site") in ("cross-site", "same-site"):
            return JSONResponse({"detail": "Cross-site request blocked"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS") and not secrets.compare_digest(
            request.headers.get("x-workbench-token", ""), token
        ):
            return JSONResponse({"detail": "Open the local dashboard first"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(OSError)
    async def file_error(request, exc):
        return JSONResponse({"detail": "Local artifact unavailable: " + str(exc)}, status_code=422)

    @app.get("/", response_class=HTMLResponse)
    def page():
        return (Path(__file__).parent / "static/index.html").read_text()

    @app.get("/app.js")
    def js():
        from fastapi.responses import Response

        return Response(
            (Path(__file__).parent / "static/app.js").read_text(),
            media_type="application/javascript",
        )

    @app.get("/api/session")
    def session():
        return {"token": token, "mode": "Local development — no login"}

    @app.get("/api/state")
    def state():
        return {
            "datasets": [
                {k: v for k, v in d.items() if k != "cases"} for d in store.list("dataset")
            ],
            "jobs": store.list("job"),
            "versions": store.list("version"),
            "active": {t: store.active_version(t) for t in ("credit_analysis", "query_plan")},
            "answers": store.list("answer"),
            "feedback": store.list("feedback"),
            "regressions": store.list("regression"),
            "comparisons": store.list("comparison"),
            "models": model_catalog(),
        }

    @app.post("/api/datasets")
    def register(request: Registration):
        data = inspect_dataset(request.path)
        # Cross-dataset frozen groups cannot reappear in training.
        for other in store.list("dataset"):
            protected = {
                c["group_id"] for c in other["cases"] if c["split"] in ("validation", "test", "oot")
            }
            if any(c["split"] == "train" and c["group_id"] in protected for c in data["cases"]):
                raise ValueError("Training overlaps a registered protected group")
            newprotected = {
                c["group_id"] for c in data["cases"] if c["split"] in ("validation", "test", "oot")
            }
            if any(c["split"] == "train" and c["group_id"] in newprotected for c in other["cases"]):
                raise ValueError("Protected group overlaps registered training data")
        return store.add("dataset", data, data["hash"])

    def compatible(item):
        if item["schema"].get("additionalProperties") is not False:
            raise ValueError("Output schema must forbid undeclared fields")
        initial = default_version(item["task"])["schema"]
        required = set(initial.get("required", []))
        if not required.issubset(item["schema"].get("required", [])):
            raise ValueError("Output schema must retain the task required fields")
        for key in required:
            if compact_json_schema(item["schema"].get("properties", {}).get(key)) != compact_json_schema(
                initial["properties"][key]
            ):
                raise ValueError(
                    "Changing a core task field requires a separate contract migration"
                )
        from jsonschema import Draft202012Validator

        for dataset in store.list("dataset"):
            if dataset["manifest"]["task"] == item["task"]:
                for case in dataset["cases"]:
                    if (
                        case["split"] in ("train", "validation")
                        and case["target"] is not None
                        and list(Draft202012Validator(item["schema"]).iter_errors(case["target"]))
                    ):
                        raise ValueError("Schema incompatible with registered development targets")

    @app.post("/api/versions")
    def save_version(request: Version):
        check_schema(request.schema_)
        if request.parent and store.get("version", request.parent)["task"] != request.task:
            raise ValueError("Parent task mismatch")
        return store.add("version", request.model_dump(by_alias=True))

    @app.post("/api/versions/{identity}/activate")
    def activate(identity: str):
        item = store.get("version", identity)
        compatible(item)
        store.active_version(item["task"], identity)
        return {
            "active_version": identity,
            "note": "Existing runs retain their frozen versions; new runs require preflight.",
        }

    @app.post("/api/feedback")
    def feedback(request: Feedback):
        return submit(
            store,
            request.interaction_id,
            request.submission_id,
            request.comment,
            request.correction,
            request.cause,
            request.expectations,
            request.semantic_review,
        )

    @app.get("/api/feedback/batch/{task}")
    def batch(task: str):
        return batch_records(store, task)

    @app.post("/api/answers")
    def import_answer(payload: dict):
        allowed = {"case", "output", "version_id", "identity", "sql_lineage"}
        if set(payload) - allowed:
            raise ValueError("Unknown answer fields")
        case = Case.model_validate(payload["case"])
        version = store.get("version", payload["version_id"])
        if case.task != version["task"]:
            raise ValueError("Task mismatch")
        lineage = payload.get("sql_lineage") or case.sql_lineage
        if lineage:
            # Allow only review-packet metadata; parameter values are always masked.
            lineage = {
                k: v
                for k, v in lineage.items()
                if k
                in (
                    "parameterised_sql",
                    "parameter_values",
                    "query_hash",
                    "schema_registry_version",
                    "snapshot_sha256",
                    "tables",
                    "selected_columns",
                    "grain",
                    "joins",
                    "filters",
                    "governed_calculations",
                    "validation",
                )
            }
            if "parameter_values" in lineage:
                lineage["parameter_values"] = ["***MASKED***"] * len(lineage["parameter_values"])
        safe_case = case.model_dump()
        safe_case["sql_lineage"] = lineage
        return store.add(
            "answer",
            {
                "case": safe_case,
                "output": payload["output"],
                "version_id": version["id"],
                "identity": payload.get(
                    "identity", {"source": "imported; not a measured model run"}
                ),
                "sql_lineage": lineage,
            },
        )

    def generation_for(profile):
        return (
            {
                "profile": "deterministic",
                "temperature": 0,
                "top_p": 0,
                "top_k": 0,
                "seed_sequence": [42, 43, 44],
                "max_tokens": 1024,
                "repeats": 3,
                "enable_thinking": False,
            }
            if profile == "deterministic"
            else {
                "profile": "serving",
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 20,
                "seed_sequence": [42, 43, 44],
                "max_tokens": 1024,
                "repeats": 3,
                "enable_thinking": False,
            }
        )

    def attach_checkpoint(spec, training_job, checkpoint):
        adapter = PROJECT / "adapters/candidates" / training_job["spec"]["task"] / training_job["id"]
        completion_file = adapter / "completion.json"
        if not completion_file.is_file():
            raise ValueError("Training run has no completion manifest")
        completion = json.loads(completion_file.read_text())
        if completion.get("status") not in ("completed", "early_stopped"):
            raise ValueError("Training did not complete successfully")
        name = "best_adapters.safetensors" if checkpoint == "best" else "adapters.safetensors"
        hash_key = "checkpoint_sha256" if checkpoint == "best" else "final_checkpoint_sha256"
        if checkpoint == "best" and not completion.get("best_optimizer_updates"):
            raise ValueError("No verified best checkpoint improved over baseline; select final explicitly")
        file = adapter / name
        if not file.is_file() or not completion.get(hash_key):
            raise ValueError("Requested checkpoint does not exist")
        checkpoint_hash = hashlib.sha256(file.read_bytes()).hexdigest()
        if checkpoint_hash != completion[hash_key]:
            raise ValueError("Requested checkpoint differs from its completion manifest")
        selected = Path(training_job["spec"]["output"]) / ("selected-" + checkpoint)
        selected.mkdir(exist_ok=True)
        selected_adapter = selected / "adapters.safetensors"
        if not selected_adapter.exists():
            shutil.copyfile(adapter / "adapter_config.json", selected / "adapter_config.json")
            shutil.copyfile(file, selected_adapter)
        if hashlib.sha256(selected_adapter.read_bytes()).hexdigest() != checkpoint_hash:
            raise ValueError("Selected checkpoint copy changed")
        spec.update(
            adapter_path=str(selected),
            checkpoint_sha256=checkpoint_hash,
            adapter_job_id=training_job["id"],
        )
        return checkpoint_hash

    def spec_for(request):
        model = next((m for m in model_catalog() if m["id"] == request.model_id), None)
        if not model:
            raise ValueError("Select a cached local 9B base snapshot")
        version = store.get("version", request.version_id)
        compatible(version)
        if version["task"] != request.task:
            raise ValueError("Version task mismatch")
        config = request.config.model_dump()
        if any(
            config[k] <= 0 or config[k] % config["grad_accumulation_steps"]
            for k in ("steps_per_eval", "save_every", "steps_per_report")
        ):
            raise ValueError("Report, validation and save intervals must align with accumulation")
        if config["optimizer"] == "adam" and config["weight_decay"]:
            raise ValueError("Weight decay requires AdamW")
        if config["schedule"] == "constant" and config["warmup_ratio"]:
            raise ValueError("Warm-up requires the cosine schedule")
        if config["batch_size"] * config["grad_accumulation_steps"] > 32:
            raise ValueError("Effective batch size is limited to 32 on this local profile")
        generation = generation_for(request.generation_profile)
        spec = {
            "kind": request.kind,
            "task": request.task,
            "version": version,
            "model": model,
            "config": config,
            "splits": request.splits,
            "generation": generation,
            "context_limit": max(4096, config["max_seq_length"]),
        }
        if request.kind == "regression":
            latest = {f["interaction_id"]: f["id"] for f in store.list("feedback")}
            spec["cases"] = [
                r["case"]
                for r in store.list("regression")
                if r["case"]["task"] == request.task
                and latest.get(r["source_interaction"]) == r["source_feedback"]
            ]
            spec["splits"] = ["development"]
            if not spec["cases"]:
                raise ValueError("No development checks with expected results")
        else:
            if not request.dataset_id:
                raise ValueError("Register a phase-2 dataset first")
            dataset = store.get("dataset", request.dataset_id)
            current = inspect_dataset(dataset["path"])
            if current["hash"] != dataset["hash"]:
                raise ValueError("Registered dataset changed")
            if dataset["manifest"]["task"] != request.task:
                raise ValueError("Dataset task mismatch")
            spec["dataset"] = dataset
        if request.adapter_job_id and request.kind == "train":
            raise ValueError("Training starts a fresh task adapter; select Base model")
        if request.adapter_job_id:
            job = store.get("job", request.adapter_job_id)
            if (
                job["status"] != "completed"
                or job["spec"]["kind"] != "train"
                or job["spec"]["task"] != request.task
            ):
                raise ValueError("Select a completed training run for this task")
            if job["spec"]["model"]["id"] != model["id"]:
                raise ValueError("Adapter base snapshot mismatch")
            attach_checkpoint(spec, job, request.checkpoint)
        return spec

    @app.post("/api/comparison-runs")
    def create_comparison(request: ComparisonRunRequest):
        if not request.splits or len(set(request.splits)) != len(request.splits):
            raise ValueError("Select one or more unique evaluation splits")
        training = store.get("job", request.training_job_id)
        if training["status"] != "completed" or training["spec"].get("kind") != "train":
            raise ValueError("Select a completed training run")
        frozen = training["spec"]
        dataset = frozen.get("dataset")
        if not dataset:
            raise ValueError("Training run has no registered dataset")
        current = inspect_dataset(dataset["path"])
        if current["hash"] != dataset["hash"]:
            raise ValueError("Registered dataset changed after training")
        available = {m["id"]: m for m in model_catalog()}
        if frozen["model"]["id"] not in available:
            raise ValueError("Training base model is no longer cached")
        model = available[frozen["model"]["id"]]
        version = frozen["version"]
        compatible(version)
        selected_cases = [
            Case.model_validate(case).model_dump()
            for case in dataset["cases"]
            if case["split"] in request.splits
        ]
        if not selected_cases:
            raise ValueError("No cases in selected evaluation splits")
        generation = generation_for(request.generation_profile)
        comparison_id = uuid4().hex
        compatibility = {
            "dataset_manifest": dataset["hash"],
            "case_set_hash": digest(selected_cases),
            "model_revision": model["id"],
            "prompt_hash": digest(version["prompt"]),
            "schema_hash": digest(version["schema"]),
            "generation_hash": digest(generation),
            "evaluator_version": EVALUATOR_VERSION,
        }
        request_hash = digest(
            {
                "training_job_id": training["id"],
                "checkpoint": request.checkpoint,
                "splits": request.splits,
                "generation": generation,
                **compatibility,
            }
        )
        active = {"queued", "running", "stopping"}
        for existing in store.list("comparison"):
            if existing["request_hash"] != request_hash:
                continue
            pair = [store.get("job", existing[key]) for key in ("base_job_id", "candidate_job_id")]
            if any(job["status"] in active for job in pair):
                raise ValueError("An identical comparison is already active")
        common = {
            "kind": "evaluate",
            "task": frozen["task"],
            "version": version,
            "model": model,
            "config": deepcopy(frozen["config"]),
            "splits": request.splits,
            "generation": generation,
            "context_limit": max(4096, frozen["config"]["max_seq_length"]),
            "dataset": dataset,
            "comparison_id": comparison_id,
            "source_training_job_id": training["id"],
            "comparison_checkpoint": request.checkpoint,
            "comparison_compatibility": compatibility,
        }
        candidate = deepcopy(common)
        checkpoint_hash = attach_checkpoint(candidate, training, request.checkpoint)
        common["comparison_checkpoint_sha256"] = checkpoint_hash
        candidate["comparison_checkpoint_sha256"] = checkpoint_hash
        base = jobs.enqueue({**common, "comparison_role": "base"})
        candidate_job = jobs.enqueue({**candidate, "comparison_role": "candidate"})
        comparison = store.add(
            "comparison",
            {
                "training_job_id": training["id"],
                "base_job_id": base["id"],
                "candidate_job_id": candidate_job["id"],
                "checkpoint": request.checkpoint,
                "splits": request.splits,
                "generation_profile": request.generation_profile,
                "request_hash": request_hash,
                "compatibility": compatibility,
            },
            comparison_id,
        )
        return {
            "comparison_id": comparison["id"],
            "base_evaluation_job_id": base["id"],
            "candidate_evaluation_job_id": candidate_job["id"],
            "status": "queued",
        }

    @app.post("/api/preflight")
    def preflight(request: JobRequest):
        spec = spec_for(request)
        spec["output"] = str(store.root / "preflight-placeholder")
        if request.kind == "train":
            _, metadata = prepare_training(spec)
            return {"passed": True, **metadata}
        cases = spec.get("cases") or spec["dataset"]["cases"]
        count = sum(c["split"] in spec["splits"] for c in cases)
        if not count:
            raise ValueError("No cases in selected evaluation splits")
        return {
            "passed": True,
            "cases": count,
            "note": "Model context checks run per case during evaluation; missing targets yield Not evaluated metrics.",
        }

    @app.post("/api/jobs")
    def start_job(request: JobRequest):
        spec = spec_for(request)
        if request.kind == "train":
            spec["output"] = str(store.root / "preflight-placeholder")
            prepare_training(spec)
        return jobs.enqueue(spec)

    @app.post("/api/jobs/{identity}/stop")
    def stop(identity: str):
        return jobs.stop(identity)

    @app.get("/api/jobs/{identity}")
    def job_detail(identity: str):
        job = store.get("job", identity)
        output = Path(job["spec"]["output"])
        result = (
            json.loads((output / "result.json").read_text())
            if (output / "result.json").exists()
            else None
        )
        adapter = PROJECT / "adapters/candidates" / job["spec"]["task"] / identity
        metrics = []
        if (adapter / "metrics.jsonl").is_file():
            for line in (adapter / "metrics.jsonl").read_text().splitlines():
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        total = current = None
        if job["spec"]["kind"] == "train":
            training_config = output / "training.yaml"
            if training_config.is_file():
                total = int(yaml.safe_load(training_config.read_text())["iters"])
            for entry in metrics:
                for kind in ("train", "validation"):
                    if kind in entry and "iteration" in entry[kind]:
                        current = max(current or 0, int(entry[kind]["iteration"]))
            current = current or 0
            if job["status"] == "completed" and total is not None:
                current = total
        else:
            progress_file = output / "progress.json"
            if progress_file.is_file():
                evaluation_progress = json.loads(progress_file.read_text())
                current = int(evaluation_progress["current_cases"])
                total = int(evaluation_progress["total_cases"])
            else:
                cases = job["spec"].get("cases") or job["spec"].get("dataset", {}).get(
                    "cases", []
                )
                total = sum(case.get("split") in job["spec"].get("splits", []) for case in cases)
                current = total if job["status"] == "completed" else 0
        accumulation = int(job["spec"].get("config", {}).get("grad_accumulation_steps", 1))
        progress = {
            "current_micro_batches": current,
            "total_micro_batches": total,
            "current_optimizer_updates": current // accumulation if current is not None else None,
            "total_optimizer_updates": total // accumulation if total is not None else None,
            "percent": round(100 * current / total, 1) if total else None,
        }
        if job["spec"]["kind"] != "train":
            progress = {
                "current_cases": current,
                "total_cases": total,
                "percent": round(100 * current / total, 1) if total else None,
            }
        return {
            "job": job,
            "result": result,
            "metrics": metrics,
            "progress": progress,
            "log": (output / "job.log").read_text(errors="replace")[-20000:]
            if (output / "job.log").exists()
            else "",
        }

    @app.get("/api/jobs/{identity}/log")
    def download_job_log(identity: str):
        job = store.get("job", identity)
        output = Path(job["spec"]["output"]).resolve()
        runs = (store.root / "runs").resolve()
        if not output.is_relative_to(runs):
            raise ValueError("Job output is outside the workspace")
        log = output / "job.log"
        if not log.is_file():
            raise ValueError("Job log is not available yet")
        return FileResponse(log, media_type="text/plain", filename=f"{identity}.log")

    @app.get("/api/compare/{left}/{right}")
    def comparison(left: str, right: str, mode: Literal["model", "prompt"] = "model"):
        a = job_detail(left)["result"]
        b = job_detail(right)["result"]
        if not a or not b or "splits" not in a or "splits" not in b:
            raise ValueError("Two completed evaluations required")
        return {"mode": mode, "metrics": compare(a, b, mode)}

    @app.get("/api/comparison-runs/{identity}")
    def comparison_run(identity: str):
        record = store.get("comparison", identity)
        base = job_detail(record["base_job_id"])
        candidate = job_detail(record["candidate_job_id"])
        states = {base["job"]["status"], candidate["job"]["status"]}
        terminal_failure = {"failed", "cancelled", "interrupted"}
        if states & terminal_failure:
            status = "failed"
        elif states == {"completed"}:
            status = "completed"
        elif "running" in states or "stopping" in states:
            status = "running"
        else:
            status = "queued"
        metrics = None
        compatibility_error = None
        if status == "completed":
            try:
                metrics = compare(base["result"], candidate["result"], "model")
            except (KeyError, TypeError, ValueError) as exc:
                status = "incompatible"
                compatibility_error = str(exc)
        return {
            "comparison": record,
            "status": status,
            "base": base,
            "candidate": candidate,
            "metrics": metrics,
            "compatibility_error": compatibility_error,
        }

    return app


def main():
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--workspace", type=Path, default=PROJECT / "outputs/workbench")
    args = p.parse_args()
    uvicorn.run(create_app(args.workspace), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
