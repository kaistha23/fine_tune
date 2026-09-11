"""Transactional, single-use SQL approvals tied to immutable request snapshots."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class ReviewConflict(ValueError):
    pass


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute(
            "CREATE TABLE IF NOT EXISTS revisions (id TEXT PRIMARY KEY, owner TEXT NOT NULL, binding TEXT NOT NULL, plan TEXT NOT NULL, packet TEXT NOT NULL, status TEXT NOT NULL, parent TEXT, created TEXT NOT NULL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS records (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, identity TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, revision TEXT, reviewer TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL)"
        )
        try:
            with con:
                yield con
        finally:
            con.close()

    def event(self, con, revision, owner, event, payload):
        con.execute(
            "INSERT INTO events(revision,reviewer,event,payload,created) VALUES (?,?,?,?,?)",
            (revision, owner, event, canonical(payload), datetime.now(UTC).isoformat()),
        )

    def create(self, owner, binding, plan, packet, parent=None):
        rid = uuid4().hex
        packet = {**packet, "revision_id": rid, "status": "pending"}
        with self.connect() as con:
            con.execute(
                "INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?)",
                (
                    rid,
                    owner,
                    binding,
                    canonical(plan),
                    canonical(packet),
                    "pending",
                    parent,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self.event(con, rid, owner, "prepared", {"parent": parent, "binding": binding})
        return packet

    def get(self, rid, owner):
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM revisions WHERE id=? AND owner=?", (rid, owner)
            ).fetchone()
        if row is None:
            raise ReviewConflict("Unknown review revision")
        return dict(row)

    def decide(self, rid, owner, decision, comment, correction=None):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT * FROM revisions WHERE id=? AND owner=?", (rid, owner)
            ).fetchone()
            if row is None or row["status"] not in (
                {"pending", "approved"} if correction is not None else {"pending"}
            ):
                raise ReviewConflict("Review is missing or no longer pending")
            status = "rejected" if correction is not None else decision
            con.execute("UPDATE revisions SET status=? WHERE id=?", (status, rid))
            self.event(
                con, rid, owner, status, {"comment": comment, "corrected_query_plan": correction}
            )
            from credit_risk.schemas import FeedbackRecord

            plan = json.loads(row["plan"])
            packet = json.loads(row["packet"])
            record = FeedbackRecord(
                interaction_id=rid,
                reviewer_id=owner,
                model_id="n/a",
                adapter_version="n/a",
                dataset_version=packet["snapshot_sha256"],
                portfolio=plan["portfolio"],
                task_type="sql_review",
                input_case_id=plan.get("obligor_id") or "portfolio-cohort",
                original_output=packet["parameterised_sql"],
                error_labels=["sql_review_rejected"] if status == "rejected" else [],
                root_cause="query",
                query_hash=packet["query_hash"],
                sql_review_status=status,
                sql_review_comment=comment,
                attempted_query_plan=correction,
                sql_review_packet=packet,
                reviewed_query_plan=plan,
                eligible_for_training=False,
            )
            con.execute(
                "INSERT INTO records(kind,identity,payload) VALUES (?,?,?)",
                ("feedback", rid, record.model_dump_json()),
            )

    def consume(self, rid, owner, binding):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT * FROM revisions WHERE id=? AND owner=?", (rid, owner)
            ).fetchone()
            if (
                row is None
                or row["status"] != "approved"
                or not hmac.compare_digest(row["binding"], binding)
            ):
                raise ReviewConflict(
                    "Approval missing, consumed, or request/schema/snapshot changed"
                )
            con.execute("UPDATE revisions SET status='consumed' WHERE id=?", (rid,))
            self.event(con, rid, owner, "consumed", {"binding": binding})

    def failed_correction(self, rid, owner, failures):
        with self.connect() as con:
            self.event(con, rid, owner, "correction_validation_failed", {"failures": failures})

    def validated_correction(self, rid, owner, plan, packet):
        """Append validated correction lineage without rewriting the original decision."""
        from credit_risk.schemas import FeedbackRecord

        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT payload FROM records WHERE kind='feedback' AND identity=? ORDER BY seq DESC LIMIT 1",
                (rid,),
            ).fetchone()
            record = FeedbackRecord.model_validate_json(row["payload"])
            record.corrected_query_plan = plan
            con.execute(
                "INSERT INTO records(kind,identity,payload) VALUES (?,?,?)",
                ("feedback", rid, record.model_dump_json()),
            )
            self.event(
                con,
                rid,
                owner,
                "correction_validated",
                {
                    "new_revision_id": packet["revision_id"],
                    "request_digest": packet["request_digest"],
                },
            )
