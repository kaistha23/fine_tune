"""Live Ask → Answer pipeline for the workbench.

Server side (CPU, no model): record the question, validate the confirmed plan, run it against
the append-only source snapshot, build the factsheet, compute the answer-memory keys and
reuse an identical earlier answer. Model side (session item, single model lane): draft the
plan, retrieve evidence, evaluate policy rules, generate three deterministic repeats, assess
them and store the answer with its lineage.

Answers from Ask are development records (split ``development``, group = obligor). They
flow into Answers & feedback exactly like evaluation answers.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from importlib import metadata
from pathlib import Path

from credit_risk.guardrails import INJECTION_PATTERNS
from credit_risk.review_store import digest
from credit_risk.schemas import EntityLevel, QueryPlan
from credit_risk.workbench import memory
from credit_risk.workbench.contracts import Case, messages

DETERMINISTIC_GENERATION = {
    "profile": "deterministic",
    "temperature": 0,
    "top_p": 0,
    "top_k": 0,
    "seed_sequence": [42, 43, 44],
    "max_tokens": 1024,
    "repeats": 3,
    "enable_thinking": False,
}
# Live answers use the full structured contract (conclusions, driver details, recommendation),
# which does not fit the 1,024-token evaluation budget sized for short targets. The budget
# matches the served API client; decoding stays greedy with the same seed sequence.
ASK_GENERATION = {**DETERMINISTIC_GENERATION, "profile": "ask-deterministic", "max_tokens": 2500}
ROLES = ("credit_analyst", "senior_credit_officer", "regulator_liaison")
ACTIVE_ITEMS = ("queued", "running")


class AskError(ValueError):
    pass


def now():
    return datetime.now(UTC).isoformat()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_versions() -> dict:
    versions = {}
    for package in ("mlx", "mlx-lm", "transformers"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def check_question(text: str) -> str:
    question = " ".join(str(text).split())
    if not 3 <= len(question) <= 2000:
        raise AskError("Ask a question of 3 to 2,000 characters")
    lowered = question.lower()
    if any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS):
        raise AskError("Question rejected by the input guardrail (prompt_injection_detected)")
    return question


def canonical_plan(plan: QueryPlan) -> dict:
    data = plan.model_dump(mode="json")
    data["metrics"] = sorted(data["metrics"])
    data["group_by"] = sorted(data["group_by"])
    return data


def parse_json_object(output):
    if isinstance(output, dict):
        return output
    text = str(output or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


# -- plan drafting --------------------------------------------------------------------------
def plan_case(question: dict, registry) -> Case:
    """The query-plan prompt context: allowed metrics and analysis types, never data rows."""
    metrics = sorted(
        name
        for name, spec in registry.data["metrics"].items()
        if question["portfolio"] in spec.get("portfolios", [])
    )
    return Case(
        case_id="ask-plan-" + question["id"],
        group_id="ask-" + question["id"],
        task="query_plan",
        split="development",
        question=question["question"],
        portfolio=question["portfolio"],
        jurisdiction=question["jurisdiction"],
        as_of_date=question["as_of_date"],
        facts={
            "instruction": "Draft one QueryPlan for this question. Use only the listed metrics.",
            "portfolio": question["portfolio"],
            "jurisdiction": question["jurisdiction"],
            "as_of_date": question["as_of_date"],
            "entity_levels": ["obligor", "facility"],
            "analysis_types": [
                "credit_deterioration",
                "ews_analysis",
                "policy_qa",
                "email_draft",
                "factsheet",
            ],
            "available_metrics": metrics,
            "maximum_months": registry.data["query_controls"]["maximum_months"],
            "rules": (
                "Name one obligor_id (entity_level obligor) or one facility_id (entity_level "
                "facility). group_by and cohort_aggregation apply only to portfolio cohorts, so "
                "leave group_by empty. date_to must not be after as_of_date; the range may span "
                "at most maximum_months."
            ),
        },
        provenance={
            "classification": "synthetic",
            "source": "workbench-ask",
            "question_id": question["id"],
            "source_snapshot_hash": question["snapshot"]["chain_hash"],
            "source_load_id": question["snapshot"]["load_id"],
            "template_family": "ask-plan",
            "transformations": ["workbench-ask-plan"],
        },
    )


def validate_plan(question: dict, plan_payload) -> QueryPlan:
    from pydantic import ValidationError

    try:
        plan = QueryPlan.model_validate(parse_json_object(plan_payload))
    except ValidationError as exc:
        problems = "; ".join(
            ".".join(str(part) for part in error["loc"]) + ": " + error["msg"] if error["loc"] else error["msg"]
            for error in exc.errors()
        )
        raise AskError("Plan is not a valid QueryPlan: " + problems) from exc
    except (ValueError, TypeError) as exc:
        raise AskError("Plan is not valid JSON: " + str(exc).splitlines()[0]) from exc
    if plan.entity_level == EntityLevel.PORTFOLIO:
        raise AskError("Ask answers one obligor or facility; cohort plans are not supported")
    if plan.jurisdiction.value != question["jurisdiction"]:
        raise AskError("Plan jurisdiction must match the question's jurisdiction")
    if plan.portfolio.value != question["portfolio"]:
        raise AskError("Plan portfolio must match the question's portfolio")
    if plan.as_of_date.isoformat() != question["as_of_date"]:
        raise AskError("Plan as_of_date must match the question's as-of date")
    if plan.date_to > datetime.now(UTC).date():
        raise AskError("Plan observes future dates")
    return plan


# -- context --------------------------------------------------------------------------------
def visible_documents(library, question: dict, retrieval_policy) -> list[dict]:
    """Document versions a retrieval for this question is allowed to consider."""
    as_of = date.fromisoformat(question["as_of_date"])
    levels = set(retrieval_policy.levels_for_role(question["role"]))
    allowed = set(retrieval_policy.data.get("approval_status_allowed", []))
    documents = library.store.list("document")
    windows = library.effective_windows(documents)
    visible = []
    for record in documents:
        window = windows[record["id"]]
        start = date.fromisoformat(window["effective_from"])
        end = date.fromisoformat(window["effective_to"]) if window["effective_to"] else None
        if (
            record["jurisdiction"] == question["jurisdiction"]
            and record["approval_status"] in allowed
            and record["confidentiality_level"] in levels
            and start <= as_of
            and (end is None or end >= as_of)
            and (not record.get("portfolio") or question["portfolio"] in record["portfolio"])
            and (not record.get("allowed_roles") or question["role"] in record["allowed_roles"])
        ):
            visible.append(
                {"id": record["id"], "file_sha256": record["file_sha256"], "window": window}
            )
    return sorted(visible, key=lambda item: item["id"])


def fact_records(sheet: dict, case_id: str) -> list[dict]:
    effective = sheet["as_of_date"]
    records = []
    for name, value in sorted(sheet.get("current_position", {}).items()):
        if isinstance(value, (bool, int, float, str)) or value is None:
            records.append(
                {
                    "fact_id": f"{case_id}-position-{name}",
                    "metric": name,
                    "value": value,
                    "unit": None,
                    "currency": None,
                    "effective_date": effective,
                    "source_id": case_id,
                }
            )
    for name, metric in sorted(sheet.get("calculated_metrics", {}).items()):
        records.append(
            {
                "fact_id": f"{case_id}-metric-{name}",
                "metric": name,
                "value": metric.get("value"),
                "unit": metric.get("unit"),
                "currency": None,
                "effective_date": effective,
                "source_id": case_id,
            }
        )
    return records[:60]


def prepare(
    question: dict,
    plan: QueryPlan,
    *,
    source_db,
    registry,
    library,
    retrieval_policy,
    rules_path: Path,
    answer_model: dict,
    answer_version: dict,
    embedder_signature: str | None,
    plan_draft: dict | None,
) -> dict:
    """Run the confirmed plan at the question's snapshot and freeze everything the model sees."""
    from credit_risk import data_service
    from credit_risk.factsheet import FactsheetError, build_factsheet
    from credit_risk.query_guard import GuardedQueryCompiler, QueryGuardError

    controls = registry.data["query_controls"]
    snapshot = question["snapshot"]
    try:
        compiled = GuardedQueryCompiler(registry).compile(plan, snapshot_load_id=snapshot["load_id"])
        rows = data_service._execute_with_limits(compiled, controls, database_path=source_db.path)
        validation = data_service.validate_result(rows, compiled, plan, controls)
    except (QueryGuardError, data_service.ResultValidationError) as exc:
        raise AskError("Plan rejected by the query guard: " + str(exc)) from exc
    if not rows:
        raise AskError("No rows for this obligor in the plan's date range at this snapshot and as-of date")
    canonical = canonical_plan(plan)
    case_id = f"ASK-{plan.obligor_id}-{plan.as_of_date.isoformat()}-{digest(canonical)[:12]}"
    try:
        sheet = build_factsheet(rows, plan, case_id=case_id).model_dump(mode="json")
    except FactsheetError as exc:
        raise AskError("Factsheet could not be built: " + str(exc)) from exc
    packet = data_service.build_sql_review_packet(compiled)
    packet.update(validation=validation, snapshot=snapshot, rows=len(rows))
    documents = visible_documents(library, question, retrieval_policy)
    context = {
        "plan": canonical,
        "snapshot": snapshot,
        "documents": documents,
        "retrieval": {
            "policy_sha256": file_hash(library.retrieval_policy),
            "embedder": embedder_signature,
        },
        "policy_rules": file_hash(rules_path),
        "version": {
            "id": answer_version["id"],
            "prompt": digest(answer_version["prompt"]),
            "schema": digest(answer_version["schema"]),
        },
        "model": {
            "model_id": answer_model["model"]["id"],
            "checkpoint_sha256": answer_model.get("checkpoint_sha256"),
        },
        "generation": ASK_GENERATION,
        "runtime": runtime_versions(),
        "role": question["role"],
    }
    context_hash = memory.context_key(context)
    keys = {
        "context_key": context_hash,
        "answer_key": memory.answer_key(context_hash, question["question"]),
        "question_key": memory.question_key(question["question"], canonical),
    }
    edited = plan_draft is not None and plan_draft != canonical
    return {
        "question": question,
        "plan": canonical,
        "plan_draft": plan_draft,
        "plan_edited": edited,
        "factsheet": sheet,
        "sql_lineage": packet,
        "snapshot": snapshot,
        "documents": documents,
        "context": context,
        "keys": keys,
        "answer_version_id": answer_version["id"],
        "model": {
            "model": answer_model["model"],
            "adapter_path": answer_model.get("adapter_path"),
            "adapter_job_id": answer_model.get("adapter_job_id"),
            "checkpoint_sha256": answer_model.get("checkpoint_sha256"),
        },
    }


# -- answer ---------------------------------------------------------------------------------
def build_case(prepared: dict, evidence: list[dict], rule_evaluations: list[dict]) -> Case:
    sheet = prepared["factsheet"]
    plan = prepared["plan"]
    question = prepared["question"]
    task_type = plan["analysis_type"] if plan["analysis_type"] != "factsheet" else "factsheet_qa"
    return Case(
        case_id=sheet["case_id"],
        group_id=plan["obligor_id"],
        task="credit_analysis",
        task_type=task_type,
        split="development",
        question=question["question"],
        portfolio=question["portfolio"],
        jurisdiction=question["jurisdiction"],
        as_of_date=question["as_of_date"],
        facts=sheet,
        fact_records=fact_records(sheet, sheet["case_id"]),
        evidence=evidence,
        rule_evaluations=rule_evaluations,
        consistency_paths=["answer_status", "risk_drivers", "recommendation"],
        sql_lineage=prepared["sql_lineage"],
        provenance={
            "classification": "synthetic",
            "source": "workbench-ask",
            "question_id": question["id"],
            "source_snapshot_hash": prepared["snapshot"]["chain_hash"],
            "source_load_id": prepared["snapshot"]["load_id"],
            "template_family": "ask-live",
            "transformations": ["workbench-ask", "factsheet-from-governed-calculators"],
            "documents": prepared["documents"],
            "plan_draft": prepared.get("plan_draft"),
            "plan_edited": prepared.get("plan_edited"),
        },
    )


PRECEDENT_SIMILARITY = 0.92


def find_precedent(store, keys: dict, vector, fields) -> dict | None:
    """A differently worded question answered in the exact same context, if one is close."""
    from credit_risk.rag.embedding import cosine

    best = None
    for record in memory.latest(store, context_key=keys["context_key"]):
        if (
            record["question_key"] == keys["question_key"]
            or record["status"] not in memory.REUSABLE
            or not record.get("question_vector")
        ):
            continue
        similarity = cosine(vector, record["question_vector"])
        if similarity >= PRECEDENT_SIMILARITY and (best is None or similarity > best["similarity"]):
            best = {
                "answer_id": record["answer_id"],
                "question": record["question"],
                "similarity": round(similarity, 4),
                "field_changes": memory.field_diff(record["fields"], fields),
            }
    return best


def complete_answer(store, item: dict, generate, retrieve, rules, embed=None):
    """Run inside the session: retrieval, rules, three repeats, assessment, storage.

    ``generate(messages, seed)`` returns text; ``retrieve(question, context)`` returns the
    retriever result or raises; ``rules(sheet, evidence)`` returns rule evaluation dicts.
    """
    from credit_risk.schemas import CreditFactsheet, Evidence
    from credit_risk.rag.schemas import AccessContext
    from credit_risk.workbench.evaluation import assess

    prepared = item["prepared"]
    question = prepared["question"]
    keys = prepared["keys"]
    reused = memory.lookup(store, keys["answer_key"])
    if reused:
        return finish_with_memory(store, question["id"], reused, prepared)
    version = store.get("version", prepared["answer_version_id"])
    started = now()
    retrieval_status = "not_run"
    evidence = []
    try:
        result = retrieve(
            question["question"],
            AccessContext(
                jurisdiction=question["jurisdiction"],
                role=question["role"],
                as_of_date=question["as_of_date"],
                portfolio=question["portfolio"],
            ),
        )
        evidence = [e.model_dump(mode="json") for e in result["evidence"]]
        retrieval_status = "evidence" if result["answer_status"] == "ANSWERED" else "insufficient"
    except Exception as exc:  # noqa: BLE001 — no documents or no index is recorded, not fatal.
        retrieval_status = "unavailable: " + str(exc).splitlines()[0]
    rule_results = rules(
        CreditFactsheet.model_validate(prepared["factsheet"]),
        [Evidence.model_validate(e) for e in evidence],
    )
    blocked = [
        f"mandatory_rule_unevaluable:{r['rule_id']}:{r['reason']}"
        for r in rule_results
        if r["mandatory"] and r["status"] == "unevaluable"
    ]
    case = build_case(prepared, evidence, rule_results)
    identity = {
        **prepared["model"],
        "version_id": version["id"],
        "prompt_hash": digest(version["prompt"]),
        "schema_hash": digest(version["schema"]),
        "generation": ASK_GENERATION,
        "source": "workbench-ask",
    }
    if blocked:
        output = {
            "answer_status": "INSUFFICIENT_EVIDENCE",
            "executive_summary": "A mandatory policy rule could not be evaluated from approved evidence.",
            "missing_information": blocked,
            "human_approval_required": True,
        }
        attempts = [{"output": output, "fields": memory.consistency_fields(output), "failures": blocked}]
        status, stable = "system", True
    else:
        attempts = []
        for seed in ASK_GENERATION["seed_sequence"]:
            text = generate(messages(case, version), seed)
            metrics, failures, parsed = assess(case, text, version)
            if parsed is None and looks_truncated(text):
                failures = [*failures, "output_truncated_at_token_budget"]
            fields = memory.consistency_fields(parsed["answer"]) if parsed else None
            attempts.append({"output": text, "metrics": metrics, "failures": failures, "fields": fields})
        if all(attempt["fields"] is None for attempt in attempts):
            # Repeats that all fail the contract are an invalid answer, not an unstable one.
            stable = len({str(attempt["output"]) for attempt in attempts}) == 1
            status = "invalid"
        else:
            stable = attempts[0]["fields"] is not None and all(
                attempt["fields"] == attempts[0]["fields"] for attempt in attempts
            )
            status = "model" if stable else "unstable"
    first = attempts[0]
    vector = None
    precedent = None
    if embed is not None:
        try:
            vector = embed(memory.normalise_question(question["question"]))
            precedent = find_precedent(store, keys, vector, first["fields"])
        except Exception:  # noqa: BLE001 — precedent lookup is advisory; the answer stands.
            vector = precedent = None
    answer = store.add(
        "answer",
        {
            "case": case.model_dump(mode="json"),
            "output": first["output"],
            "version_id": version["id"],
            "schema_hash": digest(version["schema"]),
            "prompt_hash": digest(version["prompt"]),
            "identity": identity,
            "sql_lineage": prepared["sql_lineage"],
            "metrics": first.get("metrics", {}),
            "failures": first.get("failures", []),
            "job_id": item.get("session_job_id"),
            "job_kind": "ask",
            "question_id": question["id"],
            "memory_status": status,
            "stable": stable,
            "attempt_fields": [attempt["fields"] for attempt in attempts],
            "retrieval_status": retrieval_status,
            "snapshot": prepared["snapshot"],
            "documents": prepared["documents"],
            "keys": keys,
            "precedent": precedent,
            "started_at": started,
            "finished_at": now(),
        },
    )
    memory.remember(
        store,
        answer_id=answer["id"],
        status=status,
        keys=keys,
        fields=first["fields"],
        context_summary=prepared["context"],
        question=question["question"],
        reason={"invalid": "all_repeats_failed_contract", "unstable": "repeats_disagree"}.get(status),
        question_vector=vector,
    )
    badge = {"invalid": "invalid", "unstable": "unstable"}.get(status, "new")
    if badge == "new" and precedent and precedent["field_changes"]:
        badge = "inconsistent"
    return finish(store, question["id"], answer, prepared, badge, precedent=precedent)


# -- phase 4: plan answers and verified memory --------------------------------------------
def record_plan_answer(store, question: dict, confirmed: dict, registry) -> dict | None:
    """Keep the model's plan draft as a query-plan answer labelled with the confirmed plan."""
    draft = question.get("plan_draft") or question.get("plan_draft_candidate") or question.get("plan_draft_raw")
    if draft is None:
        return None
    version = store.get("version", question["plan_version_id"])
    case = plan_case(question, registry).model_copy(update={"expected": {"query_plan": confirmed}})
    from credit_risk.workbench.evaluation import assess

    metrics, failures, _parsed = assess(case, draft, version)
    return store.add(
        "answer",
        {
            "case": case.model_dump(mode="json"),
            "output": draft,
            "version_id": version["id"],
            "schema_hash": digest(version["schema"]),
            "prompt_hash": digest(version["prompt"]),
            "identity": {**question["plan_model"], "source": "workbench-ask-plan"},
            "sql_lineage": None,
            "metrics": metrics,
            "failures": failures,
            "job_kind": "ask_plan",
            "question_id": question["id"],
            "plan_edited": draft != confirmed,
        },
    )


def verify(store, answer: dict, *, output=None, reason: str, source_answer=None) -> dict:
    """Make a human-confirmed or corrected answer the one returned for this exact key."""
    if not answer.get("keys"):
        raise AskError("Only Ask answers carry the keys needed for answer memory")
    record = answer
    if output is not None or source_answer is not None:
        chosen = output if output is not None else source_answer["output"]
        record = store.add(
            "answer",
            {
                **{k: v for k, v in answer.items() if k not in ("id", "created_at")},
                "output": chosen,
                "identity": {**answer["identity"], "source": reason},
                "memory_status": "verified",
                "stable": True,
                "attempt_fields": None,
                "verified_from": source_answer["id"] if source_answer else answer["id"],
                "failures": [],
                "metrics": {},
            },
        )
    parsed = _parsed(record)
    memory.remember(
        store,
        answer_id=record["id"],
        status="verified",
        keys=answer["keys"],
        fields=memory.consistency_fields(parsed),
        context_summary=answer.get("context") or _context_of(store, answer),
        question=answer["case"]["question"],
        reason=reason,
    )
    return record


def _context_of(store, answer):
    found = memory.latest(store, answer_id=answer["id"])
    return found[-1]["context"] if found else {}


def looks_truncated(text) -> bool:
    body = str(text or "").strip()
    return body.startswith(("{", "```")) and not body.rstrip("`").rstrip().endswith("}")


def finish_with_memory(store, question_id, record, prepared):
    answer = store.get("answer", record["answer_id"])
    badge = "verified" if record["status"] == "verified" else "reused"
    return finish(store, question_id, answer, prepared, badge, memory_record=record)


def finish(store, question_id, answer, prepared, badge, memory_record=None, precedent=None):
    keys = prepared["keys"]
    earlier = memory.previous(store, keys["question_key"], keys["context_key"])
    comparison = None
    if earlier and badge in ("new", "unstable"):
        comparison = {
            "previous_answer_id": earlier["answer_id"],
            "changed_context": memory.context_changes(earlier["context"], prepared["context"]),
            "field_changes": memory.field_diff(earlier["fields"], memory.consistency_fields(_parsed(answer))),
        }
    return store.update(
        "question",
        question_id,
        status="answered",
        answer_id=answer["id"],
        badge="changed" if badge == "new" and comparison and comparison["field_changes"] else badge,
        memory_record_id=memory_record["id"] if memory_record else None,
        comparison=comparison,
        precedent=precedent,
        answered_at=now(),
        error=None,
    )


def _parsed(answer):
    try:
        return parse_json_object(answer["output"])
    except (ValueError, TypeError):
        return None
