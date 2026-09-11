"""Transactional local state, immutable submissions and versions, mutable job status."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "workbench.sqlite3"
        with self.connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS items (kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind,id))"
            )
            con.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=15)
        try:
            with con:
                yield con
        finally:
            con.close()

    def add(self, kind, payload, identity=None):
        identity = identity or uuid4().hex
        record = {**payload, "id": identity, "created_at": datetime.now(UTC).isoformat()}
        with self.connect() as con:
            old = con.execute(
                "SELECT payload FROM items WHERE kind=? AND id=?", (kind, identity)
            ).fetchone()
            if old:
                old = json.loads(old[0])
                if {k: v for k, v in old.items() if k not in ("id", "created_at")} != payload:
                    raise ValueError("Submission ID already used with different content")
                return old
            con.execute(
                "INSERT INTO items VALUES (?,?,?)",
                (kind, identity, json.dumps(record, allow_nan=False)),
            )
        return record

    def list(self, kind):
        with self.connect() as con:
            rows = con.execute(
                "SELECT payload FROM items WHERE kind=? ORDER BY rowid", (kind,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get(self, kind, identity):
        with self.connect() as con:
            row = con.execute(
                "SELECT payload FROM items WHERE kind=? AND id=?", (kind, identity)
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown {kind} ID")
        return json.loads(row[0])

    def update_job(self, identity, **values):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT payload FROM items WHERE kind='job' AND id=?", (identity,)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown job")
            job = {**json.loads(row[0]), **values}
            con.execute(
                "UPDATE items SET payload=? WHERE kind='job' AND id=?", (json.dumps(job), identity)
            )
        return job

    def active_version(self, task, version=None):
        with self.connect() as con:
            if version is not None:
                con.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (task, version))
            row = con.execute("SELECT value FROM settings WHERE key=?", (task,)).fetchone()
        return row[0] if row else None
