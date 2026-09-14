"""Local dashboard. Run: uv run python -m credit_risk.workbench.server"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import duckdb
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from credit_risk.schemas import compact_json_schema
from credit_risk.workbench.contracts import Case, default_version, inspect_dataset
from credit_risk.review_store import digest
from credit_risk.workbench.evaluation import (
    EVALUATOR_VERSION,
    LOWER_IS_BETTER,
    MIN_REPORTABLE_SLICE,
    check_schema,
    compare,
)
from credit_risk.workbench.feedback import batch_records, submit
from credit_risk.workbench.jobs import Jobs
from credit_risk.workbench import ask, memory
from credit_risk.workbench.ask import ASK_GENERATION, DETERMINISTIC_GENERATION
from credit_risk.workbench.documents import MAX_DOCUMENT_BYTES, DocumentLibrary, DocumentMetadata
from credit_risk.workbench.sources import MAX_UPLOAD_BYTES, SourceDatabase
from credit_risk.workbench.store import Store
from credit_risk.workbench.worker import PROJECT, embedding_catalog, model_catalog, prepare_training


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


class SourceInitialization(Strict):
    source_path: str | None = None


class SourceAppend(Strict):
    staged_id: str = Field(min_length=1, max_length=128)
    table: str = Field(min_length=1, max_length=128)
    file_name: str | None = Field(default=None, max_length=256)


class DocumentRegistration(Strict):
    staged_id: str = Field(min_length=1, max_length=128)
    file_name: str = Field(min_length=1, max_length=256)
    metadata: DocumentMetadata


class ModelChoice(Strict):
    model_id: str
    adapter_job_id: str | None = None
    checkpoint: Literal["best", "final"] = "best"


class QuestionRequest(Strict):
    question: str = Field(min_length=3, max_length=2000)
    jurisdiction: Literal["SAMA", "CBUAE"]
    portfolio: Literal["retail", "sme", "corporate"]
    as_of_date: str
    role: Literal["credit_analyst", "senior_credit_officer", "regulator_liaison"] = "credit_analyst"
    snapshot_load_id: int | None = Field(default=None, ge=1)
    plan_model: ModelChoice
    plan_version_id: str
    answer_model: ModelChoice
    answer_version_id: str
    draft_plan: bool = True


class PlanConfirmation(Strict):
    plan: dict


class AskFeedback(Strict):
    comment: str = Field(default="", max_length=8000)


class DocumentIndexRequest(Strict):
    embedder_id: str | None = None


CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def embedder_signature(model):
    """MLXEmbedder's signature without loading weights: model path and hidden size."""
    config = json.loads((Path(model["path"]) / "config.json").read_text())
    return f"{model['path']}:{int(config['hidden_size'])}"


class ReplayRequest(Strict):
    model: "ModelChoice"


class DatasetBuildRequest(Strict):
    base_dataset_id: str = Field(min_length=1, max_length=128)
    dataset_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    feedback_fraction: float = Field(default=0.2, gt=0, lt=1)
    paraphrases: dict[str, list[str]] = Field(default_factory=dict)


class ComparisonRunRequest(Strict):
    training_job_id: str = Field(min_length=1, max_length=128)
    reference_job_id: str | None = Field(default=None, max_length=128)
    checkpoint: Literal["best", "final"] = "best"
    generation_profile: Literal["deterministic", "serving"] = "deterministic"
    splits: list[Literal["validation", "test", "oot"]] = Field(
        default_factory=lambda: ["validation", "test", "oot"]
    )


def create_app(root=None, start_scheduler=True):
    store = Store(root or PROJECT / "outputs/workbench")
    jobs = Jobs(store)
    token = secrets.token_urlsafe(32)
    recommended_ids = {}
    for task in ("credit_analysis", "query_plan"):
        recommended = default_version(task)
        if store.active_version(task) is None:
            item = store.add("version", recommended)
            store.active_version(task, item["id"])
        match = next(
            (
                version
                for version in store.list("version")
                if version["task"] == task
                and version["prompt"] == recommended["prompt"]
                and version["schema"] == recommended["schema"]
            ),
            None,
        )
        if match is None:
            prefix = "Recommended structured contract v"
            numbers = [
                int(version["name"][len(prefix) :])
                for version in store.list("version")
                if version["task"] == task
                and version["name"].startswith(prefix)
                and version["name"][len(prefix) :].isdigit()
            ]
            recommended.update(
                name=prefix + str(max(numbers, default=2) + 1),
                parent=store.active_version(task),
            )
            match = store.add("version", recommended)
        recommended_ids[task] = match["id"]

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

    @app.exception_handler(duckdb.Error)
    async def database_error(request, exc):
        return JSONResponse({"detail": "Source database error: " + str(exc).splitlines()[0]}, status_code=422)

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

    def moment(value):
        try:
            return datetime.fromisoformat(value) if value else None
        except (TypeError, ValueError):
            return None

    def job_times(job):
        """Recorded start/finish; jobs from before timing fall back to their log file times."""
        started, finished = moment(job.get("started_at")), moment(job.get("finished_at"))
        terminal = job["status"] not in ("queued", "running", "stopping")
        output = job["spec"].get("output")
        log = Path(output) / "job.log" if output else None
        if log and log.is_file() and (started is None or (terminal and finished is None)):
            stat = log.stat()
            if started is None:
                born = getattr(stat, "st_birthtime", stat.st_ctime)
                started = datetime.fromtimestamp(born, UTC)
            if terminal and finished is None:
                finished = datetime.fromtimestamp(stat.st_mtime, UTC)
        return started, finished

    def timing(job, current, total, unit, anchor=None, last=None):
        """Elapsed time and a rate-based remaining-time estimate.

        The anchor is the first measured point after model loading (baseline validation
        or evaluation start), so the one-off load does not inflate the per-unit rate.
        """
        started, finished = job_times(job)
        clock = datetime.now(UTC)
        end = finished or clock
        result = {
            "started_at": started.isoformat() if started else None,
            "finished_at": finished.isoformat() if finished else None,
            "elapsed_seconds": round((end - started).total_seconds()) if started else None,
            "seconds_per_unit": None,
            "eta_seconds": None,
            "unit": unit,
        }
        anchor = anchor or started
        last = last or end
        if anchor and current and total and last > anchor:
            per = (last - anchor).total_seconds() / current
            result["seconds_per_unit"] = round(per, 1)
            if job["status"] in ("running", "stopping") and current < total:
                waited = (clock - last).total_seconds()
                result["eta_seconds"] = round(max(0.0, per * (total - current) - waited))
        return result

    def without_cases(job):
        """Job specs freeze full datasets; the dashboard list only needs their summaries."""
        spec = job["spec"]
        slim = {k: v for k, v in spec.items() if k != "cases"}
        if spec.get("cases") is not None:
            slim["case_count"] = len(spec["cases"])
        if spec.get("dataset"):
            slim["dataset"] = {k: v for k, v in spec["dataset"].items() if k != "cases"}
        started, finished = job_times(job)
        return {
            **job,
            "spec": slim,
            "started_at": started.isoformat() if started else None,
            "finished_at": finished.isoformat() if finished else None,
        }

    source_database = {}

    def sources():
        # Loaded on first use so a registry problem surfaces on the Source data panel
        # instead of stopping the whole dashboard.
        if "value" not in source_database:
            from credit_risk.query_guard import SchemaRegistry
            from credit_risk.settings import settings

            source_database["value"] = SourceDatabase(
                store.root / "source",
                SchemaRegistry(CONFIGS / "schema_registry.yaml", settings.schema_registry_version),
            )
        return source_database["value"]

    @app.get("/api/sources")
    def source_summary():
        return sources().summary()

    @app.post("/api/sources/initialize")
    def initialize_sources(request: SourceInitialization):
        path = Path(request.source_path) if request.source_path else PROJECT / "data/curated/credit_risk.duckdb"
        return sources().initialize(path.expanduser())

    @app.post("/api/sources/stage")
    async def stage_source(request: Request, table: str, filename: str):
        declared = int(request.headers.get("content-length") or 0)
        if declared > MAX_UPLOAD_BYTES:
            raise ValueError("Uploaded file exceeds 200 MB")
        staged = sources().stage(await request.body(), filename)
        return {**sources().validate(staged["staged_id"], table), "file_name": staged["file_name"]}

    @app.post("/api/sources/append")
    def append_source(request: SourceAppend):
        return sources().append(request.staged_id, request.table, request.file_name)

    library = DocumentLibrary(store, store.root / "documents", CONFIGS / "retrieval.yaml")

    def default_embedder(identity=None):
        catalog = embedding_catalog()
        model = next((m for m in catalog if identity in (None, m["id"])), None)
        if model is None:
            raise ValueError("Select a cached local Qwen3-Embedding snapshot")
        return model

    @app.get("/api/documents")
    def documents():
        catalog = embedding_catalog()
        signature = embedder_signature(catalog[0]) if catalog else None
        items = library.list()
        for item in items:
            item["indexed"] = signature in item["indexed_with"] if signature else False
        return {"documents": items, "embedders": catalog, "signature": signature}

    @app.post("/api/documents/stage")
    async def stage_document(request: Request, filename: str):
        if int(request.headers.get("content-length") or 0) > MAX_DOCUMENT_BYTES:
            raise ValueError("Uploaded document exceeds 50 MB")
        return library.stage(await request.body(), filename)

    @app.post("/api/documents")
    def register_document(request: DocumentRegistration):
        return library.register(request.staged_id, request.file_name, request.metadata)

    @app.post("/api/documents/index")
    def index_documents(request: DocumentIndexRequest):
        model = default_embedder(request.embedder_id)
        pending = library.pending(embedder_signature(model))
        if not pending:
            raise ValueError("Every registered document is already indexed with this embedder")
        active = {"queued", "running", "stopping"}
        if any(j["spec"].get("kind") == "index_documents" and j["status"] in active for j in store.list("job")):
            raise ValueError("A document indexing job is already queued or running")
        return jobs.enqueue(
            {
                "kind": "index_documents",
                "task": "documents",
                "embedder": model,
                "documents_root": str(library.root.resolve()),
                "retrieval_policy": str(library.retrieval_policy),
                "documents": [item["id"] for item in pending],
            }
        )

    @app.get("/api/state")
    def state():
        return {
            "datasets": [
                {k: v for k, v in d.items() if k != "cases"} for d in store.list("dataset")
            ],
            "jobs": [without_cases(job) for job in store.list("job")],
            "versions": store.list("version"),
            "active": {t: store.active_version(t) for t in ("credit_analysis", "query_plan")},
            "recommended": recommended_ids,
            "preflights": store.list("preflight"),
            "evaluation_policy": {
                "min_reportable_slice": MIN_REPORTABLE_SLICE,
                "lower_is_better": sorted(LOWER_IS_BETTER),
            },
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
        record = submit(
            store,
            request.interaction_id,
            request.submission_id,
            request.comment,
            request.correction,
            request.cause,
            request.expectations,
            request.semantic_review,
        )
        answer = store.get("answer", request.interaction_id)
        if record["correction_valid"] and answer.get("keys") and answer["case"]["task"] == "credit_analysis":
            # A valid correction to an Ask answer is returned for identical questions at once;
            # training on it still waits for the normal eligibility and dataset steps.
            verified = ask.verify(store, answer, output=record["correction"], reason="human correction")
            record = {**record, "verified_answer_id": verified["id"]}
        return record

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
            deepcopy(DETERMINISTIC_GENERATION)
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
        accumulation = config["grad_accumulation_steps"]
        for key in ("steps_per_eval", "save_every", "steps_per_report"):
            # Defaults follow the chosen accumulation; explicit values are still validated.
            if key not in request.config.model_fields_set:
                config[key] = -(-config[key] // accumulation) * accumulation
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
                "reference_job_id": request.reference_job_id,
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
        if request.reference_job_id:
            # Compare against the previous adapter instead of the base: same frozen cases,
            # prompt/schema and generation; only the adapter differs.
            reference = store.get("job", request.reference_job_id)
            if (
                reference["status"] != "completed"
                or reference["spec"].get("kind") != "train"
                or reference["spec"]["task"] != frozen["task"]
                or reference["id"] == training["id"]
            ):
                raise ValueError("Reference must be another completed training run for this task")
            if reference["spec"]["model"]["id"] != model["id"]:
                raise ValueError("Reference adapter uses a different base snapshot")
            attach_checkpoint(common, reference, "best")
        common["comparison_checkpoint_sha256"] = checkpoint_hash
        candidate["comparison_checkpoint_sha256"] = checkpoint_hash
        base = jobs.enqueue({**common, "comparison_role": "base"})
        candidate_job = jobs.enqueue({**candidate, "comparison_role": "candidate"})
        comparison = store.add(
            "comparison",
            {
                "training_job_id": training["id"],
                "reference_job_id": request.reference_job_id,
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
            lengths = sorted(metadata["token_lengths"])
            summary = {
                "dataset_id": request.dataset_id,
                "version_id": request.version_id,
                "model_id": request.model_id,
                "max_seq_length": spec["config"]["max_seq_length"],
                "chat_template_hash": metadata["chat_template_hash"],
                "examples": len(lengths),
                "max_tokens": metadata["max_tokens"],
                "p95_tokens": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))],
                "min_assistant_tokens": metadata["min_assistant_tokens"],
            }
            store.add("preflight", summary, digest(summary))
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
        anchor = last = None
        if job["spec"]["kind"] == "train":
            training_config = output / "training.yaml"
            if training_config.is_file():
                total = int(yaml.safe_load(training_config.read_text())["iters"])
            for entry in metrics:
                for kind in ("train", "validation"):
                    if kind in entry and "iteration" in entry[kind]:
                        current = max(current or 0, int(entry[kind]["iteration"]))
                        reported = moment(entry[kind].get("reported_at"))
                        if reported:
                            anchor = anchor or reported
                            last = reported
            current = current or 0
            if job["status"] == "completed" and total is not None:
                current = total
        else:
            progress_file = output / "progress.json"
            if progress_file.is_file():
                evaluation_progress = json.loads(progress_file.read_text())
                current = int(evaluation_progress["current_cases"])
                total = int(evaluation_progress["total_cases"])
                anchor = moment(evaluation_progress.get("started_at"))
                last = moment(evaluation_progress.get("updated_at"))
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
        is_train = job["spec"]["kind"] == "train"
        job_timing = timing(
            job,
            current,
            total,
            "micro-batch" if is_train else {"index_documents": "document", "replay": "answer"}.get(job["spec"]["kind"], "case"),
            # Training metrics start with the iteration-0 baseline, so rate counts from there.
            anchor,
            last,
        )
        return {
            "job": job,
            "result": result,
            "metrics": metrics,
            "progress": progress,
            "timing": job_timing,
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
        remaining, estimable = 0, True
        per_case = base["timing"]["seconds_per_unit"] or candidate["timing"]["seconds_per_unit"]
        for side in (base, candidate):
            if side["job"]["status"] in ("running", "stopping"):
                if side["timing"]["eta_seconds"] is None:
                    estimable = False
                else:
                    remaining += side["timing"]["eta_seconds"]
            elif side["job"]["status"] == "queued":
                # A queued side has no rate yet; the other side's per-case time is the estimate.
                if per_case and side["progress"]["total_cases"]:
                    remaining += round(per_case * side["progress"]["total_cases"])
                else:
                    estimable = False
        pair_timing = {
            "elapsed_seconds": sum(
                side["timing"]["elapsed_seconds"] or 0 for side in (base, candidate)
            ),
            "eta_seconds": remaining if estimable and status in ("queued", "running") else None,
        }
        for side in (base, candidate):
            # Logs are fetched per job; polling the pair should not resend both.
            side.pop("log", None)
            side["job"] = without_cases(side["job"])
        gates = None
        if status == "completed":
            from credit_risk.workbench.learning import consistency_gates, load_gates

            gates = consistency_gates(
                store,
                load_gates(CONFIGS / "consistency_gates.yaml"),
                candidate,
                record["training_job_id"],
                candidate["job"]["spec"].get("checkpoint_sha256"),
            )
        return {
            "comparison": record,
            "status": status,
            "consistency_gates": gates,
            "base": base,
            "candidate": candidate,
            "metrics": metrics,
            "compatibility_error": compatibility_error,
            "timing": pair_timing,
        }

    # -- Ask -----------------------------------------------------------------------------
    def model_choice(choice: ModelChoice, task: str) -> dict:
        model = next((m for m in model_catalog() if m["id"] == choice.model_id), None)
        if not model:
            raise ValueError("Select a cached local base snapshot")
        selection = {"model": model, "adapter_path": None, "adapter_job_id": None, "checkpoint_sha256": None}
        if choice.adapter_job_id:
            job = store.get("job", choice.adapter_job_id)
            if job["status"] != "completed" or job["spec"].get("kind") != "train" or job["spec"]["task"] != task:
                raise ValueError(f"Select a completed {task} training run")
            if job["spec"]["model"]["id"] != model["id"]:
                raise ValueError("Adapter base snapshot mismatch")
            attach_checkpoint(selection, job, choice.checkpoint)
            selection["checkpoint"] = choice.checkpoint
        return selection

    def version_for(identity: str, task: str) -> dict:
        version = store.get("version", identity)
        if version["task"] != task:
            raise ValueError(f"Select a {task} prompt/schema version")
        compatible(version)
        return version

    def session_spec(selection: dict) -> dict:
        catalog = embedding_catalog()
        return {
            "kind": "session",
            "task": "ask",
            "session_key": digest(
                {"model": selection["model"]["id"], "checkpoint": selection.get("checkpoint_sha256")}
            ),
            "model": selection["model"],
            "adapter_path": selection.get("adapter_path"),
            "adapter_job_id": selection.get("adapter_job_id"),
            "checkpoint_sha256": selection.get("checkpoint_sha256"),
            "embedder": catalog[0] if catalog else None,
            "generation": deepcopy(ASK_GENERATION),
            "context_limit": 8192,
            "idle_seconds": 600,
            "documents_root": str(library.root.resolve()),
            "retrieval_policy": str(library.retrieval_policy),
            "schema_registry": str(CONFIGS / "schema_registry.yaml"),
            "policy_rules": str(CONFIGS / "policy_rules.yaml"),
        }

    def queue_item(kind: str, question: dict, selection: dict, **payload) -> dict:
        spec = session_spec(selection)
        item = store.add(
            "session_item",
            {
                "kind": kind,
                "question_id": question["id"],
                "session_key": spec["session_key"],
                "status": "queued",
                **payload,
            },
        )
        active = {"queued", "running", "stopping"}
        if not any(
            job["status"] in active and job["spec"].get("session_key") == spec["session_key"]
            for job in store.list("job")
        ):
            jobs.enqueue(spec)
        return item

    def question_view(question: dict) -> dict:
        view = dict(question)
        if question.get("answer_id"):
            view["answer"] = store.get("answer", question["answer_id"])
        view["items"] = [
            {k: v for k, v in item.items() if k != "prepared"}
            for item in store.list("session_item")
            if item["question_id"] == question["id"]
        ]
        return view

    @app.post("/api/questions")
    def ask_question(request: QuestionRequest):
        text = ask.check_question(request.question)
        try:
            as_of = datetime.fromisoformat(request.as_of_date).date()
        except ValueError as exc:
            raise ValueError("as_of_date must be YYYY-MM-DD") from exc
        if as_of > datetime.now(UTC).date():
            raise ValueError("as_of_date must not be in the future")
        snapshot = sources().snapshot(request.snapshot_load_id)
        plan_selection = model_choice(request.plan_model, "query_plan")
        model_choice(request.answer_model, "credit_analysis")  # validated now, resolved on confirm
        version_for(request.plan_version_id, "query_plan")
        version_for(request.answer_version_id, "credit_analysis")
        question = store.add(
            "question",
            {
                "question": text,
                "jurisdiction": request.jurisdiction,
                "portfolio": request.portfolio,
                "as_of_date": as_of.isoformat(),
                "role": request.role,
                "snapshot": snapshot,
                "plan_model": request.plan_model.model_dump(),
                "plan_version_id": request.plan_version_id,
                "answer_model": request.answer_model.model_dump(),
                "answer_version_id": request.answer_version_id,
                "status": "drafting_plan" if request.draft_plan else "plan_ready",
                "plan_draft": None,
                "plan_error": None,
            },
        )
        if request.draft_plan:
            queue_item("plan_draft", question, plan_selection, plan_version_id=request.plan_version_id)
        return question_view(question)

    @app.get("/api/questions")
    def list_questions():
        return [
            {k: question.get(k) for k in ("id", "question", "status", "badge", "as_of_date", "jurisdiction", "portfolio", "snapshot", "answer_id", "created_at", "error")}
            for question in reversed(store.list("question")[-200:])
        ]

    @app.get("/api/questions/{identity}")
    def get_question(identity: str):
        return question_view(store.get("question", identity))

    @app.post("/api/questions/{identity}/plan")
    def confirm_plan(identity: str, request: PlanConfirmation):
        from credit_risk.query_guard import SchemaRegistry
        from credit_risk.rag.filters import RetrievalPolicy
        from credit_risk.settings import settings

        question = store.get("question", identity)
        if question["status"] not in ("plan_ready", "failed"):
            raise ValueError(f"Question is {question['status']}; a plan can be confirmed once it is ready")
        plan = ask.validate_plan(question, request.plan)
        answer_selection = model_choice(ModelChoice(**question["answer_model"]), "credit_analysis")
        version = version_for(question["answer_version_id"], "credit_analysis")
        catalog = embedding_catalog()
        registry = SchemaRegistry(CONFIGS / "schema_registry.yaml", settings.schema_registry_version)
        prepared = ask.prepare(
            question,
            plan,
            source_db=sources(),
            registry=registry,
            library=library,
            retrieval_policy=RetrievalPolicy(library.retrieval_policy),
            rules_path=CONFIGS / "policy_rules.yaml",
            answer_model=answer_selection,
            answer_version=version,
            embedder_signature=embedder_signature(catalog[0]) if catalog else None,
            plan_draft=question.get("plan_draft"),
        )
        plan_answer = None
        if not question.get("plan_answer_id"):
            plan_answer = ask.record_plan_answer(store, question, prepared["plan"], registry)
        question = store.update(
            "question", identity, status="answering", plan_confirmed=prepared["plan"],
            plan_edited=bool(plan_answer and plan_answer["plan_edited"]) or prepared["plan_edited"],
            plan_answer_id=plan_answer["id"] if plan_answer else question.get("plan_answer_id"),
            keys=prepared["keys"], error=None,
        )
        reused = memory.lookup(store, prepared["keys"]["answer_key"])
        if reused:
            question = ask.finish_with_memory(store, identity, reused, prepared)
        else:
            queue_item("answer", question, answer_selection, prepared=prepared)
        return question_view(question)

    @app.post("/api/questions/{identity}/plan-feedback")
    def plan_feedback(identity: str, request: AskFeedback):
        question = store.get("question", identity)
        if not question.get("plan_answer_id") or not question.get("plan_confirmed"):
            raise ValueError("Run a model-drafted plan first; there is no plan draft to give feedback on")
        edited = question.get("plan_edited")
        comment = request.comment.strip() or (
            "The drafted plan needed changes before it could run." if edited else "Plan draft feedback."
        )
        return submit(
            store,
            question["plan_answer_id"],
            "plan-" + uuid4().hex,
            comment,
            question["plan_confirmed"] if edited else None,
            "model_behaviour" if edited else "unknown",
        )

    @app.post("/api/answers/{identity}/verify")
    def verify_answer(identity: str, request: AskFeedback):
        answer = store.get("answer", identity)
        if answer.get("memory_status") in ("unstable", "invalid"):
            raise ValueError("Unstable or invalid answers cannot be confirmed; submit a correction instead")
        record = ask.verify(store, answer, reason="confirmed correct in Ask")
        submit(store, identity, "verify-" + uuid4().hex, "Confirmed correct. " + request.comment.strip())
        return {"verified_answer_id": record["id"], "note": "Identical questions in this context now return this answer."}

    @app.post("/api/questions/{identity}/still-valid")
    def still_valid(identity: str, request: AskFeedback):
        question = store.get("question", identity)
        comparison = question.get("comparison") or {}
        if not comparison.get("previous_answer_id") or not question.get("answer_id"):
            raise ValueError("Only an answer that changed since an earlier answer can be re-confirmed")
        current = store.get("answer", question["answer_id"])
        previous = store.get("answer", comparison["previous_answer_id"])
        record = ask.verify(store, current, source_answer=previous, reason="earlier answer re-confirmed as still valid")
        submit(store, current["id"], "still-valid-" + uuid4().hex, "Earlier answer still valid in the new context. " + request.comment.strip())
        store.update("question", identity, badge="verified", answer_id=record["id"])
        return question_view(store.get("question", identity))

    @app.post("/api/replay")
    def start_replay(request: ReplayRequest):
        from credit_risk.workbench.learning import verified_answers

        answers = [record["answer_id"] for record in verified_answers(store)]
        if not answers:
            raise ValueError("No verified answers yet; confirm or correct Ask answers first")
        selection = model_choice(request.model, "credit_analysis")
        return jobs.enqueue(
            {
                "kind": "replay",
                "task": "credit_analysis",
                "model": selection["model"],
                "adapter_path": selection.get("adapter_path"),
                "adapter_job_id": selection.get("adapter_job_id"),
                "checkpoint_sha256": selection.get("checkpoint_sha256"),
                "generation": deepcopy(ASK_GENERATION),
                "context_limit": 8192,
                "answers": answers,
            }
        )

    @app.get("/api/verified")
    def verified():
        from credit_risk.workbench.learning import verified_answers

        return [
            {"answer_id": r["answer_id"], "question": r["question"], "reason": r["reason"], "fields": r["fields"], "created_at": r["created_at"]}
            for r in verified_answers(store)
        ]

    @app.post("/api/datasets/build")
    def build_dataset_version(request: DatasetBuildRequest):
        from credit_risk.workbench.learning import build_dataset

        return build_dataset(
            store,
            store.get("dataset", request.base_dataset_id),
            request.dataset_version,
            PROJECT / "data/workbench",
            request.paraphrases,
            request.feedback_fraction,
        )

    @app.get("/api/sessions")
    def sessions():
        active = {"queued", "running", "stopping"}
        items = store.list("session_item")
        return [
            {
                "job_id": job["id"],
                "status": job["status"],
                "model": job["spec"]["model"]["label"],
                "adapter_job_id": job["spec"].get("adapter_job_id"),
                "queued_items": sum(i["session_key"] == job["spec"]["session_key"] and i["status"] == "queued" for i in items),
                "releasing": (Path(job["spec"]["output"]) / "release").exists(),
                "started_at": job_times(job)[0].isoformat() if job_times(job)[0] else None,
            }
            for job in store.list("job")
            if job["spec"].get("kind") == "session" and job["status"] in active
        ]

    @app.post("/api/sessions/release")
    def release_sessions():
        released = []
        for job in store.list("job"):
            if job["spec"].get("kind") == "session" and job["status"] in ("queued", "running"):
                if job["status"] == "queued":
                    jobs.stop(job["id"])
                else:
                    (Path(job["spec"]["output"]) / "release").write_text("released\n")
                released.append(job["id"])
        return {"released": released, "note": "Running sessions finish their current step, then free the model lane."}

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
