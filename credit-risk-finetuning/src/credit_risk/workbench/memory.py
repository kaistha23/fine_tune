"""Answer memory: the same question in the same context returns the same answer.

A fine-tuned adapter does not remember answers; it learns how to answer. Consistency comes
from identical inputs, deterministic generation and this store:

* ``context_key`` hashes everything the model saw: confirmed plan, data snapshot, visible
  document versions, policy rules, prompt/schema version, model and checkpoint, generation
  settings and runtime versions.
* ``answer_key`` adds the normalised question.

Records are append-only. A status change (``model``, ``unstable``, ``system``, ``verified``,
``superseded``) is a new record for the same answer; the latest one wins. Consistency is
compared on structured fields, never on prose.
"""

from __future__ import annotations

import re

from credit_risk.review_store import digest

REUSABLE = {"model", "system", "verified"}
# An invalid answer is returned again for an identical context (regenerating would repeat it),
# but it is never a precedent or a baseline for "changed since last answer".
EXACT_REUSE = REUSABLE | {"invalid"}
_STAGE = re.compile(r"\bstage\b[^0-9;]{0,30}?\b([1-3])\b", re.IGNORECASE)
_ENTITY = re.compile(r"\b(?:OBL|FAC)[-_ ]?\d+(?:[-_ ]\d+)?\b", re.IGNORECASE)
_MIGRATION = re.compile(r"\b(?:migrat\w*|from\s+(?:stage\s+)?[1-3]\s+to)\b", re.IGNORECASE)
_IDENTIFIER = re.compile(r"\b(obl|fac)[-_ ]?(\d+)(?:[-_ ](\d+))?\b", re.IGNORECASE)


def normalise_question(text: str) -> str:
    """Case, whitespace, trailing punctuation and identifier spelling do not change a question."""

    def identifier(match):
        prefix, number, suffix = match.group(1).upper(), int(match.group(2)), match.group(3)
        return f"{prefix}-{number:04d}" + (f"-{int(suffix)}" if suffix else "")

    return _IDENTIFIER.sub(identifier, " ".join(text.split()).rstrip("?.! ").casefold())


def consistency_fields(answer) -> dict | None:
    """Stable decision fields of a credit answer; None when the output is not a valid object."""
    if not isinstance(answer, dict):
        return None
    cited = sorted(
        {
            evidence_id
            for group in ("facts", "conclusions", "risk_driver_details")
            for item in answer.get(group) or []
            if isinstance(item, dict)
            for evidence_id in item.get("evidence_ids") or []
        }
    )
    stage = next(
        (
            item.get("value")
            for item in answer.get("conclusions") or []
            if isinstance(item, dict) and "stage" in str(item.get("conclusion_type", "")).lower()
        ),
        None,
    )
    if stage is None:
        # Facts state the stage as a short claim: "current_position.stage = 2", "The current
        # credit stage for OBL-0002 is 2". Migration statements ("from 1 to 2") are skipped,
        # and claims about the current position are preferred.
        statements = [
            _ENTITY.sub(" ", str(item.get("statement", "")))
            for item in answer.get("facts") or []
            if isinstance(item, dict)
        ]
        statements = [text for text in statements if not _MIGRATION.search(text)]
        statements.sort(key=lambda text: "current" not in text.lower())
        for text in statements:
            found = _STAGE.search(text)
            if found:
                stage = int(found.group(1))
                break
    return {
        "answer_status": answer.get("answer_status"),
        "stage": stage,
        "risk_drivers": sorted(" ".join(str(d).lower().split()) for d in answer.get("risk_drivers") or []),
        "recommendation": " ".join(str(answer.get("recommendation") or "").lower().split()),
        "cited_evidence_ids": cited,
    }


def context_key(context: dict) -> str:
    return digest(context)


def answer_key(context_hash: str, question: str) -> str:
    return digest({"context_key": context_hash, "question": normalise_question(question)})


def question_key(question: str, plan: dict) -> str:
    """Same question about the same plan, whatever the snapshot, documents or model."""
    return digest({"question": normalise_question(question), "plan": plan})


def latest(store, **match) -> list[dict]:
    """Latest memory record per answer, filtered on exact field values."""
    current = {}
    for record in store.list("answer_memory"):
        current[record["answer_id"]] = record
    return [
        record
        for record in current.values()
        if all(record.get(field) == value for field, value in match.items())
    ]


def lookup(store, key: str) -> dict | None:
    """Verified answers win over model answers; unstable and superseded are never reused."""
    candidates = [record for record in latest(store, answer_key=key) if record["status"] in EXACT_REUSE]
    for status in ("verified", "system", "model", "invalid"):
        found = [record for record in candidates if record["status"] == status]
        if found:
            return found[-1]
    return None


def previous(store, question_hash: str, current_context: str) -> dict | None:
    """The most recent reusable answer to the same question and plan in a different context."""
    found = [
        record
        for record in latest(store, question_key=question_hash)
        if record["context_key"] != current_context and record["status"] in REUSABLE
    ]
    return found[-1] if found else None


def remember(store, *, answer_id, status, keys, fields, context_summary, question, reason=None,
             question_vector=None):
    return store.add(
        "answer_memory",
        {
            "question_vector": question_vector,
            "answer_id": answer_id,
            "status": status,
            "answer_key": keys["answer_key"],
            "context_key": keys["context_key"],
            "question_key": keys["question_key"],
            "question": normalise_question(question),
            "fields": fields,
            "context": context_summary,
            "reason": reason,
        },
    )


def field_diff(before: dict | None, after: dict | None) -> dict:
    if before is None or after is None:
        return {}
    return {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in sorted(set(before) | set(after))
        if before.get(name) != after.get(name)
    }


def context_changes(before: dict, after: dict) -> list[str]:
    """Human-readable reasons two contexts differ."""
    reasons = []
    labels = {
        "snapshot": "data snapshot (source loads)",
        "documents": "visible document versions",
        "policy_rules": "policy rules",
        "version": "prompt/schema version",
        "model": "model or adapter checkpoint",
        "generation": "generation settings",
        "runtime": "runtime library versions",
        "retrieval": "retrieval policy or embedder",
    }
    for name, label in labels.items():
        if before.get(name) != after.get(name):
            reasons.append(label)
    return reasons
