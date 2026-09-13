"""Deterministic field-based metrics and real-provider consistency harness."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict

from jsonschema import Draft202012Validator

from credit_risk.guardrails import validate_output
from credit_risk.review_store import digest
from credit_risk.schemas import CreditResponse, Evidence, QueryPlan

MIN_REPORTABLE_SLICE = 10
BOOTSTRAP_SAMPLES = 1000
EVALUATOR_VERSION = "field-checks-v2"


def check_schema(schema):
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ValueError("Invalid JSON schema") from exc

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ("$ref", "$dynamicRef") and not v.startswith("#"):
                    raise ValueError("Only local schema references allowed")
                walk(v)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(schema)


def field(value, path):
    for part in path.split("."):
        if isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def normal(value):
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, list):
        return sorted((normal(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, dict):
        return {k: normal(v) for k, v in sorted(value.items())}
    return value


def plan_key(value):
    plan = QueryPlan.model_validate(value).model_dump(mode="json")
    for key in ("metrics", "group_by"):
        plan[key] = sorted(plan[key])
    return plan


def projection(case, answer):
    if case.task == "query_plan":
        return plan_key(answer)
    if not case.consistency_paths:
        return None
    values = {p: field(answer, p) for p in case.consistency_paths}
    if any(v is None for v in values.values()):
        raise ValueError("Missing consistency field")
    return normal(values)


def assess(case, answer, version, query_checker=None):
    metrics = {}
    failures = []
    calibration_observations = []
    try:
        answer = json.loads(answer) if isinstance(answer, str) else answer
        if not isinstance(answer, dict):
            raise TypeError("Expected JSON object")
        # Reject non-finite JSON numbers even if the backend parser permits them.
        json.dumps(answer, allow_nan=False)
        errors = list(Draft202012Validator(version["schema"]).iter_errors(answer))
        if errors:
            raise ValueError("Output schema mismatch")
    except (ValueError, TypeError):
        failed = {"json_validity": 0.0}
        expected = case.expected
        for path in expected.get("fields", {}):
            failed["field:" + path] = 0.0
        if expected.get("numerics"):
            failed["numerical_agreement"] = 0.0
        if case.task == "credit_analysis":
            failed["extractive_support_heuristic"] = 0.0
            if "risk_drivers" in expected:
                failed.update(driver_precision=0.0, driver_recall=0.0)
            if expected.get("evidence_ids"):
                failed["citation_recall"] = 0.0
            if expected.get("must_abstain"):
                failed["abstention_recall"] = 0.0
        else:
            failed.update(plan_validity=0.0, compilation_success=0.0)
            if expected.get("query_plan"):
                for key in plan_key(expected["query_plan"]):
                    failed["plan:" + key] = 0.0
            if "rows" in expected:
                failed["result_agreement"] = 0.0
        return failed, ["invalid_json_or_schema"], None
    metrics["json_validity"] = 1.0
    expected = case.expected
    for path, value in expected.get("fields", {}).items():
        match = normal(field(answer, path)) == normal(value)
        metrics["field:" + path] = float(match)
        if not match:
            failures.append("field:" + path)
    nums = expected.get("numerics", {})
    if nums:
        checks = []
        for path, specification in nums.items():
            if isinstance(specification, dict):
                value = specification["value"]
                absolute = specification.get("abs_tolerance", 1e-8)
                relative = specification.get("rel_tolerance", 0)
            else:
                value = specification
                absolute = 1e-8
                relative = 0
            actual = field(answer, path)
            ok = (
                type(actual) in (int, float)
                and math.isfinite(actual)
                and math.isclose(actual, value, rel_tol=relative, abs_tol=absolute)
            )
            if isinstance(specification, dict):
                for suffix in ("unit", "currency", "as_of_date"):
                    wanted = specification.get(suffix)
                    actual_path = specification.get(suffix + "_path")
                    if wanted is not None:
                        actual_path = actual_path or path.rsplit(".", 1)[0] + "." + suffix
                        ok = ok and normal(field(answer, actual_path)) == normal(wanted)
            checks.append(ok)
            if not ok:
                failures.append("numeric:" + path)
        metrics["numerical_agreement"] = sum(checks) / len(checks)
    if case.task == "credit_analysis":
        try:
            response = CreditResponse.model_validate(answer)
            evidence = [Evidence.model_validate(e) for e in case.evidence]
            check = validate_output(response, evidence, case.facts.get("case_id"), case.facts)
            metrics["extractive_support_heuristic"] = float(check.passed)
            failures.extend(check.failures)
            refs = [eid for fact in response.facts for eid in fact.evidence_ids]
            refs.extend(
                eid
                for item in [*response.conclusions, *response.risk_driver_details]
                for eid in item.evidence_ids
            )
            if response.recommendation_detail:
                refs.extend(response.recommendation_detail.rationale_evidence_ids)
            valid = {e.evidence_id for e in evidence} | {case.facts.get("case_id")}
            if refs:
                metrics["citation_resolution"] = sum(r in valid for r in refs) / len(refs)
            required = expected.get("evidence_ids")
            if required:
                metrics["citation_recall"] = len(set(required) & set(refs)) / len(set(required))
            if "risk_drivers" in expected:
                required = set(normal(expected["risk_drivers"]))
                actual = set(normal(response.risk_drivers))
                metrics["driver_recall"] = (
                    len(required & actual) / len(required) if required else float(not actual)
                )
                metrics["driver_precision"] = (
                    len(required & actual) / len(actual) if actual else float(not required)
                )
            if expected.get("must_abstain"):
                metrics["abstention_recall"] = float(
                    response.answer_status == "INSUFFICIENT_EVIDENCE"
                )
            if response.answer_status == "INSUFFICIENT_EVIDENCE":
                metrics["abstention_precision"] = float(bool(expected.get("must_abstain")))
            labels = expected.get("confidence_labels", {})
            if labels:
                squared_errors = []
                for path, outcome in labels.items():
                    confidence = field(answer, path)
                    if type(confidence) not in (int, float) or outcome not in (0, 1):
                        failures.append("confidence:" + path)
                        squared_errors.append(1.0)
                    else:
                        squared_errors.append((confidence - outcome) ** 2)
                        calibration_observations.append(
                            {"path": path, "confidence": confidence, "outcome": outcome}
                        )
                metrics["confidence_brier_score"] = sum(squared_errors) / len(squared_errors)
        except ValueError:
            failures.append("credit_contract_mismatch")
            metrics["credit_contract_validity"] = 0.0
    else:
        try:
            actual = plan_key(answer)
            metrics["plan_validity"] = 1.0
            if expected.get("query_plan"):
                wanted = plan_key(expected["query_plan"])
                for key in wanted:
                    metrics["plan:" + key] = float(actual[key] == wanted[key])
                    if actual[key] != wanted[key]:
                        failures.append("plan:" + key)
            if query_checker:
                query_metrics, lineage = query_checker(case, actual)
                metrics.update(query_metrics)
                return metrics, failures, {
                    "answer": answer,
                    "sql_lineage": lineage,
                    "calibration": calibration_observations,
                }
        except ValueError as exc:
            metrics["plan_validity"] = 0.0
            metrics["compilation_success"] = 0.0
            failures.append(str(exc))
    return metrics, failures, {
        "answer": answer,
        "sql_lineage": case.sql_lineage,
        "calibration": calibration_observations,
    }



def assess_training_target(case, answer, version, semantic_review=None):
    """Reuse field checks while keeping training admission separate from serving policy."""
    metrics, failures, parsed = assess(case, answer, version)
    if case.task == "credit_analysis" and parsed:
        from credit_risk.guardrails import is_admissible_training_target

        admission = is_admissible_training_target(
            CreditResponse.model_validate(parsed["answer"]),
            [Evidence.model_validate(e) for e in case.evidence], case.facts,
            semantic_review or case.provenance.get("semantic_review"),
        )
        metrics["training_target_admissible"] = float(admission.passed)
        if admission.passed:
            failures = [f for f in failures if f not in {"unsupported_claim", "unverified_narrative"}]
        else:
            failures.extend(admission.failures)
    return metrics, failures, parsed


def _bootstrap_interval(values, seed=42):
    if len(values) < 2:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return [means[24], means[974]]


def aggregate(rows):
    by = defaultdict(list)
    for row in rows:
        for key, value in row["metrics"].items():
            by[key].append(value)
    result = {
        k: {
            "value": sum(v) / len(v),
            "denominator": len(v),
            "ci95": _bootstrap_interval(v, seed=42 + i),
            "sufficient_sample": len(v) >= MIN_REPORTABLE_SLICE,
        }
        for i, (k, v) in enumerate(sorted(by.items()))
    }
    weighted = [
        (float(row["severity_weight"]), float(not row["failures"]))
        for row in rows
        if row.get("severity_weight") is not None
    ]
    if weighted:
        total = sum(weight for weight, _ in weighted)
        result["business_weighted_pass_rate"] = {
            "value": sum(weight * passed for weight, passed in weighted) / total,
            "denominator": len(weighted),
            "weight_total": total,
            "sufficient_sample": len(weighted) >= MIN_REPORTABLE_SLICE,
        }
    return result


def calibration_summary(rows):
    observations = [item for row in rows for item in row.get("calibration", [])]
    if not observations:
        return {"status": "Not evaluated", "denominator": 0, "bins": []}
    bins = []
    weighted_error = 0.0
    for low in (0.0, 0.2, 0.4, 0.6, 0.8):
        high = low + 0.2
        members = [
            item
            for item in observations
            if low <= item["confidence"] <= high
            and (item["confidence"] < high or high == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(item["confidence"] for item in members) / len(members)
        accuracy = sum(item["outcome"] for item in members) / len(members)
        weighted_error += len(members) * abs(mean_confidence - accuracy)
        bins.append(
            {
                "from": low,
                "to": high,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "status": "Reportable"
        if len(observations) >= MIN_REPORTABLE_SLICE
        else "Small sample",
        "denominator": len(observations),
        "brier_score": sum(
            (item["confidence"] - item["outcome"]) ** 2 for item in observations
        )
        / len(observations),
        "expected_calibration_error": weighted_error / len(observations),
        "bins": bins,
    }


def evaluate(cases, provider, version, identity, query_checker=None, progress=None):
    """Provider is called three times per case. Mocking it tests plumbing, not model accuracy."""
    rows = []
    projections = {}
    equivalent = {}
    failed_fields = []
    for case_number, case in enumerate(cases, start=1):
        attempts = []
        keys = []
        for repeat in range(3):
            try:
                output = provider(case, version)
            except Exception as exc:  # noqa: BLE001 — preserve failed cases in metrics.
                # Recorded per-case failure; never silently drop a case.
                output = None
                failed_fields.append({"case_id": case.case_id, "error": type(exc).__name__})
            metrics, failures, parsed = assess(case, output, version, query_checker)
            try:
                key = projection(case, parsed["answer"]) if parsed else None
            except ValueError:
                key = None
            keys.append(key)
            attempts.append(
                {
                    "metrics": metrics,
                    "failures": failures,
                    "output": output,
                    "sql_lineage": parsed["sql_lineage"] if parsed else case.sql_lineage,
                    "calibration": parsed.get("calibration", []) if parsed else [],
                }
            )
        first = attempts[0]
        metrics = dict(first["metrics"])
        if case.consistency_paths or case.task == "query_plan":
            metrics["repeated_agreement"] = float(
                keys[0] is not None and keys[0] == keys[1] == keys[2]
            )
        if metrics.get("repeated_agreement") == 0:
            first["failures"].append("repeated_answer_disagreement")
            first["consistency_projections"] = keys
        projections[case.case_id] = keys[0]
        if case.equivalence_id:
            family_key = (case.split, case.equivalence_id)
            context = case.context_hash()
            if family_key in equivalent and equivalent[family_key]["context"] != context:
                raise ValueError("Equivalent family context mismatch")
            equivalent.setdefault(family_key, {"context": context, "members": []})[
                "members"
            ].append((case.case_id, keys[0]))
        rows.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "portfolio": case.portfolio,
                "task": case.task,
                "severity_weight": case.expected.get("severity_weight"),
                "metrics": metrics,
                "failures": first["failures"],
                "attempts": attempts,
                "calibration": first["calibration"],
            }
        )
        if progress:
            progress(case_number, len(cases))
    for family in equivalent.values():
        members = family["members"]
        if len(members) > 1:
            consistent = members[0][1] is not None and all(k == members[0][1] for _, k in members)
            for row in rows:
                if row["case_id"] in {i for i, _ in members}:
                    row["metrics"]["equivalent_agreement"] = float(consistent)
                    if not consistent:
                        row["failures"].append("equivalent_question_disagreement")
    for case, row in zip(cases, rows, strict=True):
        if case.distinct_from:
            available = [other for other in case.distinct_from if other in projections]
            if available:
                row["metrics"]["negative_control_distinction"] = float(
                    projections[case.case_id] is not None
                    and all(
                        projections[o] is not None and projections[o] != projections[case.case_id]
                        for o in available
                    )
                )
                if row["metrics"]["negative_control_distinction"] == 0:
                    row["failures"].append("changed_question_false_match")
    from credit_risk.evaluation.cli import release_report, serializable
    from credit_risk.evaluation.gates import ReleaseGates
    from credit_risk.evaluation.metrics import GoldCase
    from credit_risk.settings import settings

    release_cases = [GoldCase(
        case_id=c.case_id, group_id=c.group_id, portfolio=c.portfolio,
        jurisdiction=c.jurisdiction, task_type=c.task, question=c.question,
        factsheet=c.facts, factsheet_case_id=c.facts.get("case_id"),
        available_evidence=[Evidence.model_validate(e) for e in c.evidence],
        must_abstain=c.expected.get("must_abstain", False),
        is_injection_attempt=c.expected.get("is_injection_attempt", False),
        required_risk_drivers=c.expected.get("risk_drivers", []),
        required_evidence_ids=c.expected.get("evidence_ids", []),
        expected_numerics={k: v["value"] if isinstance(v, dict) else v
                           for k, v in c.expected.get("numerics", {}).items()},
    ) for c in cases if c.task == "credit_analysis" and c.split in ("test", "oot")]
    release_ids = {c.case_id for c in release_cases}
    predictions = {r["case_id"]: {"output": r["attempts"][0]["output"],
                                  "provider_failed": r["attempts"][0]["output"] is None}
                   for r in rows if r["case_id"] in release_ids}
    release = release_report(release_cases, predictions, ReleaseGates(settings.evaluation_thresholds),
                             metadata={"generation": identity.get("generation"),
                                       "candidate_identity": identity,
                                       "prompt_hash": identity.get("prompt_hash", digest(version["prompt"]))})
    return {
        "identity": identity,
        "release_evaluation": serializable(release),
        "evaluator_version": EVALUATOR_VERSION,
        "case_set_hash": digest([c.model_dump() for c in cases]),
        "splits": {
            s: aggregate([r for r in rows if r["split"] == s])
            for s in ("train", "validation", "test", "oot", "development")
        },
        "portfolios": {
            p: aggregate([r for r in rows if r["portfolio"] == p])
            for p in ("retail", "sme", "corporate")
        },
        "calibration": {
            s: calibration_summary([r for r in rows if r["split"] == s])
            for s in ("train", "validation", "test", "oot", "development")
        },
        "cases": rows,
        "provider_errors": failed_fields,
        "note": "Field consistency and extractive support are not semantic expert assessment. Development feedback checks are not held-out accuracy.",
    }


def compare(left, right, mode="model"):
    for key in ("case_set_hash", "evaluator_version"):
        if left[key] != right[key]:
            raise ValueError("Incompatible benchmark inputs or evaluator")
    if left["identity"].get("generation") != right["identity"].get("generation"):
        raise ValueError("Generation settings differ")
    if mode == "model":
        if left["identity"].get("version_id") != right["identity"].get("version_id"):
            raise ValueError("Model comparisons require the same prompt/schema version")
    elif mode == "prompt":
        for key in ("model", "checkpoint_sha256", "schema_hash"):
            if (
                key not in left["identity"]
                or key not in right["identity"]
                or left["identity"][key] != right["identity"][key]
            ):
                raise ValueError(
                    "Prompt experiments require the same model, checkpoint and output schema"
                )
    else:
        raise ValueError("Unknown comparison mode")
    scorecards = {
        s: {
            m: {
                "base": v["value"],
                "candidate": right["splits"][s][m]["value"],
                "delta": right["splits"][s][m]["value"] - v["value"],
                "denominator": v["denominator"],
            }
            for m, v in metrics.items()
            if m in right["splits"][s] and v["denominator"] == right["splits"][s][m]["denominator"]
        }
        for s, metrics in left["splits"].items()
    }
    left_rows = {row["case_id"]: row for row in left["cases"]}
    right_rows = {row["case_id"]: row for row in right["cases"]}
    paired = {}
    for split in left["splits"]:
        metric_deltas = defaultdict(list)
        for identity in sorted(set(left_rows) & set(right_rows)):
            a, b = left_rows[identity], right_rows[identity]
            if a["split"] != split or b["split"] != split:
                continue
            for metric in set(a["metrics"]) & set(b["metrics"]):
                metric_deltas[metric].append(b["metrics"][metric] - a["metrics"][metric])
        paired[split] = {
            metric: {
                "mean_delta": sum(values) / len(values),
                "ci95": _bootstrap_interval(values, seed=84 + i),
                "cases": len(values),
                "wins": sum(value > 0 for value in values),
                "losses": sum(value < 0 for value in values),
                "ties": sum(value == 0 for value in values),
            }
            for i, (metric, values) in enumerate(sorted(metric_deltas.items()))
        }
    return {"scorecards": scorecards, "paired": paired}
