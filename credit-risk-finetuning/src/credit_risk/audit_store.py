"""Append-only application records in transactional SQLite, separate from analytics."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path


class AuditStore:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=15)
        con.execute(
            "CREATE TABLE IF NOT EXISTS records (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, identity TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        try:
            with con:
                yield con
        finally:
            con.close()

    def append(self, kind, identity, payload):
        with self.connect() as con:
            con.execute(
                "INSERT INTO records(kind,identity,payload) VALUES (?,?,?)",
                (kind, identity, payload),
            )

    def read(self, kind, identity=None):
        with self.connect() as con:
            if identity is None:
                rows = con.execute(
                    "SELECT payload FROM records WHERE kind=? ORDER BY seq", (kind,)
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT payload FROM records WHERE kind=? AND identity=? ORDER BY seq",
                    (kind, identity),
                ).fetchall()
        return [r[0] for r in rows]
