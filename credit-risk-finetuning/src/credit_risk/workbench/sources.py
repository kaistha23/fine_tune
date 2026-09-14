"""Append-only governed source data for the workbench.

Rows are never updated or deleted. Every accepted file becomes one numbered load, recorded
in a hash-chained ledger. A query snapshot is a load watermark: it sees loads at or below
it and resolves each grain row to its latest load, so an answer given at watermark N can be
replayed exactly after later loads (including restatements) arrive.

DuckDB allows one writing process at a time. Writes here are short, serialised by a file
lock, and every query elsewhere opens its own read-only connection.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

from credit_risk.query_guard import SchemaRegistry

FORMATS = {".parquet": "read_parquet", ".csv": "read_csv", ".jsonl": "read_json", ".json": "read_json"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
EXAMPLES = 5
LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS source_loads (
    load_id INTEGER NOT NULL,
    table_name VARCHAR NOT NULL,
    file_name VARCHAR NOT NULL,
    file_sha256 VARCHAR NOT NULL,
    row_count BIGINT NOT NULL,
    restated_rows BIGINT NOT NULL,
    min_observation_date DATE,
    max_observation_date DATE,
    max_data_cutoff_date DATE,
    registry_version VARCHAR NOT NULL,
    loaded_at VARCHAR NOT NULL,
    prev_chain_hash VARCHAR NOT NULL,
    chain_hash VARCHAR NOT NULL,
    PRIMARY KEY (load_id, table_name)
)
"""
GENESIS = "0" * 64
SQL_TYPES = {
    "integer": "BIGINT",
    "number": "DOUBLE",
    "probability": "DOUBLE",
    "boolean": "BOOLEAN",
    "date": "DATE",
    "string": "VARCHAR",
}


class SourceError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def chain(previous: str, entries: list[dict]) -> str:
    material = json.dumps(
        {
            "previous": previous,
            "entries": sorted(
                (
                    {k: entry[k] for k in ("table_name", "file_sha256", "row_count")}
                    for entry in entries
                ),
                key=lambda item: item["table_name"],
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


class SourceDatabase:
    def __init__(self, root: Path, registry: SchemaRegistry):
        self.root = Path(root)
        self.path = self.root / "credit_risk.duckdb"
        self.staging = self.root / "staging"
        self.registry = registry

    # -- helpers -------------------------------------------------------------------------
    @property
    def initialized(self) -> bool:
        return self.path.is_file()

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "source.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def tables(self) -> dict[str, dict]:
        controls = self.registry.data["query_controls"]
        return {
            name: spec
            for name, spec in self.registry.data["tables"].items()
            if name in controls["allowed_tables"]
        }

    def table_spec(self, table: str) -> dict:
        tables = self.tables()
        if table not in tables:
            raise SourceError(f"Table is not allowlisted: {table}")
        load_column = self.registry.data["point_in_time"].get("load_column")
        if not load_column or load_column not in tables[table]["allowed_columns"]:
            raise SourceError(f"Table {table} declares no load column")
        return tables[table]

    def read_only(self):
        if not self.initialized:
            raise SourceError("Source database is not initialized")
        return duckdb.connect(str(self.path), read_only=True)

    # -- state ---------------------------------------------------------------------------
    def ledger(self) -> list[dict]:
        if not self.initialized:
            return []
        with self.read_only() as con:
            cursor = con.execute("SELECT * FROM source_loads ORDER BY load_id, table_name")
            columns = [item[0] for item in cursor.description]
            return [
                {
                    k: (v.isoformat() if isinstance(v, date) else v)
                    for k, v in zip(columns, row, strict=True)
                }
                for row in cursor.fetchall()
            ]

    def snapshot(self, load_id: int | None = None) -> dict:
        entries = self.ledger()
        if not entries:
            raise SourceError("Source database has no loads")
        watermark = max(entry["load_id"] for entry in entries) if load_id is None else load_id
        at = [entry for entry in entries if entry["load_id"] == watermark]
        if not at:
            raise SourceError(f"Unknown load watermark: {watermark}")
        return {"load_id": watermark, "chain_hash": at[0]["chain_hash"]}

    def summary(self) -> dict:
        if not self.initialized:
            return {"initialized": False, "ledger": [], "snapshot": None, "tables": {}}
        ledger = self.ledger()
        with self.read_only() as con:
            counts = {
                table: con.execute(f"SELECT count(*) FROM {quoted(table)}").fetchone()[0]
                for table in self.tables()
            }
        return {
            "initialized": True,
            "ledger": ledger,
            "snapshot": self.snapshot() if ledger else None,
            "tables": counts,
            "registry_version": self.registry.version,
        }

    # -- initialization ------------------------------------------------------------------
    def initialize(self, source: Path) -> dict:
        """Seed from an existing governed DuckDB as load 1. Never overwrites."""
        source = Path(source).resolve()
        if not source.is_file():
            raise SourceError(f"Source database not found: {source}")
        with self.locked():
            if self.initialized:
                raise SourceError("Source database already initialized")
            temporary = self.path.with_suffix(".initializing")
            temporary.unlink(missing_ok=True)
            shutil.copyfile(source, temporary)
            digest = file_sha256(source)
            try:
                with duckdb.connect(str(temporary)) as con:
                    con.execute(LEDGER_DDL)
                    entries = []
                    for table, spec in self.tables().items():
                        load_column = self.registry.data["point_in_time"]["load_column"]
                        self.table_spec(table)
                        columns = {row[0] for row in con.execute(f"DESCRIBE {quoted(table)}").fetchall()}
                        expected = set(spec["allowed_columns"])
                        missing = expected - columns - {load_column}
                        if missing:
                            raise SourceError(f"{table} is missing registry columns: {sorted(missing)}")
                        if load_column not in columns:
                            con.execute(f"ALTER TABLE {quoted(table)} ADD COLUMN {quoted(load_column)} INTEGER")
                        con.execute(f"UPDATE {quoted(table)} SET {quoted(load_column)} = 1")
                        entries.append(self._entry(con, table, quoted(table), 1, source.name, digest, 0))
                    self._write_ledger(con, 1, entries, GENESIS)
                temporary.replace(self.path)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        return self.summary()

    # -- staging and validation ----------------------------------------------------------
    def stage(self, content: bytes, filename: str) -> dict:
        suffix = Path(filename).suffix.lower()
        if suffix not in FORMATS:
            raise SourceError("Upload Parquet, CSV or JSONL records")
        if not content:
            raise SourceError("Uploaded file is empty")
        if len(content) > MAX_UPLOAD_BYTES:
            raise SourceError("Uploaded file exceeds 200 MB")
        digest = hashlib.sha256(content).hexdigest()
        self.staging.mkdir(parents=True, exist_ok=True)
        target = self.staging / (digest + suffix)
        if not target.exists():
            temporary = target.with_suffix(suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
        return {"staged_id": target.name, "file_name": Path(filename).name, "file_sha256": digest}

    def staged(self, staged_id: str) -> Path:
        path = (self.staging / staged_id).resolve()
        if path.parent != self.staging.resolve() or not path.is_file():
            raise SourceError("Unknown staged file")
        return path

    def _reader(self, path: Path) -> str:
        function = FORMATS[path.suffix.lower()]
        options = ""
        if function == "read_csv":
            options = ", header = true, auto_detect = true"
        elif function == "read_json":
            options = ", format = 'newline_delimited'"
        return f"{function}('{str(path).replace(chr(39), chr(39) * 2)}'{options})"

    def validate(self, staged_id: str, table: str) -> dict:
        path = self.staged(staged_id)
        spec = self.table_spec(table)
        pit = self.registry.data["point_in_time"]
        load_column = pit["load_column"]
        definitions = {k: v for k, v in spec["allowed_columns"].items() if k != load_column}
        errors: list[dict] = []

        def fail(code, count=None, examples=None, detail=None):
            errors.append(
                {k: v for k, v in {"code": code, "rows": count, "examples": examples, "detail": detail}.items() if v is not None}
            )

        digest = file_sha256(path)
        with duckdb.connect() as con:
            try:
                con.execute(f"CREATE TABLE staged AS SELECT * FROM {self._reader(path)}")
            except duckdb.Error as exc:
                raise SourceError("File could not be read: " + str(exc).splitlines()[0]) from exc
            con.execute("SET enable_external_access=false")
            columns = [row[0] for row in con.execute("DESCRIBE staged").fetchall()]
            rows = con.execute("SELECT count(*) FROM staged").fetchone()[0]
            if not rows:
                fail("no_rows")
            missing = sorted(set(definitions) - set(columns))
            extra = sorted(set(columns) - set(definitions))
            if missing:
                fail("missing_columns", detail=missing)
            if extra:
                # Default deny: an undeclared column (including load_id) is never accepted.
                fail("undeclared_columns", detail=extra)

            def count(where, select="*"):
                total = con.execute(f"SELECT count(*) FROM staged WHERE {where}").fetchone()[0]
                sample = (
                    con.execute(f"SELECT {select} FROM staged WHERE {where} LIMIT {EXAMPLES}").fetchall()
                    if total
                    else []
                )
                return total, [list(map(_plain, row)) for row in sample]

            present = [column for column in definitions if column in columns]
            grain = list(spec.get("grain_columns", []))
            required = set(grain) | {"portfolio", "jurisdiction", pit["data_cutoff_column"]}
            if pit.get("model_run_column") in definitions:
                required.add(pit["model_run_column"])
            for column in present:
                rule = definitions[column]
                name = quoted(column)
                sql_type = SQL_TYPES.get(rule.get("type"), "VARCHAR")
                cast = f"TRY_CAST({name} AS {sql_type})"
                total, sample = count(f"{name} IS NOT NULL AND {cast} IS NULL", name)
                if total:
                    fail("invalid_type:" + column, total, sample, rule.get("type"))
                    continue
                if column in required:
                    total, _ = count(f"{name} IS NULL")
                    if total:
                        fail("missing_required:" + column, total)
                if rule.get("type") == "string":
                    total, _ = count(f"{name} IS NOT NULL AND trim(CAST({name} AS VARCHAR)) = ''")
                    if total:
                        fail("blank_string:" + column, total)
                if rule.get("type") == "integer":
                    total, sample = count(
                        f"{name} IS NOT NULL AND {cast} <> TRY_CAST({name} AS DOUBLE)", name
                    )
                    if total:
                        fail("invalid_type:" + column, total, sample, "integer")
                if "min" in rule:
                    total, sample = count(f"{cast} < {float(rule['min'])}", name)
                    if total:
                        fail("below_minimum:" + column, total, sample, rule["min"])
                if "max" in rule:
                    total, sample = count(f"{cast} > {float(rule['max'])}", name)
                    if total:
                        fail("above_maximum:" + column, total, sample, rule["max"])
                allowed = rule.get("allowed")
                if column == "portfolio":
                    allowed = spec.get("portfolios")
                if column == "jurisdiction":
                    allowed = self.registry.data["query_controls"]["allowed_jurisdictions"]
                if allowed:
                    values = ", ".join(_literal(value) for value in allowed)
                    total, sample = count(f"{name} IS NOT NULL AND {cast} NOT IN ({values})", name)
                    if total:
                        fail("not_allowed:" + column, total, sample, allowed)

            typed = not any(error["code"].startswith("invalid_type") for error in errors)
            if typed and not missing:
                observation = quoted("observation_date")
                cutoff = quoted(pit["data_cutoff_column"])
                total, sample = count(
                    f"CAST({cutoff} AS DATE) < CAST({observation} AS DATE)",
                    f"{observation}, {cutoff}",
                )
                if total:
                    fail("cutoff_before_observation", total, sample)
                if pit.get("model_run_column") in definitions:
                    model_run = quoted(pit["model_run_column"])
                    total, sample = count(
                        f"CAST({model_run} AS DATE) < CAST({observation} AS DATE)",
                        f"{observation}, {model_run}",
                    )
                    if total:
                        fail("model_run_before_observation", total, sample)
                today = datetime.now(UTC).date().isoformat()
                total, sample = count(f"CAST({cutoff} AS DATE) > DATE '{today}'", cutoff)
                if total:
                    fail("future_data_cutoff", total, sample)
                if grain:
                    keys = ", ".join(quoted(column) for column in grain)
                    duplicates = con.execute(
                        f"SELECT count(*) FROM (SELECT {keys} FROM staged GROUP BY {keys} HAVING count(*) > 1)"
                    ).fetchone()[0]
                    if duplicates:
                        fail("duplicate_grain", duplicates, detail=grain)
            window = (
                con.execute(
                    f"SELECT min(CAST(observation_date AS DATE)), max(CAST(observation_date AS DATE)),"
                    f" max(CAST({quoted(pit['data_cutoff_column'])} AS DATE)) FROM staged"
                ).fetchone()
                if typed and not missing and rows
                else (None, None, None)
            )
        restated = 0
        if not self.initialized:
            fail("source_not_initialized")
        else:
            ledger = self.ledger()
            if any(entry["file_sha256"] == digest for entry in ledger):
                fail("file_already_loaded")
            if typed and not missing and grain and not errors:
                restated = self._restated(path, table, grain)
        return {
            "staged_id": staged_id,
            "table": table,
            "file_sha256": digest,
            "rows": rows,
            "restated_rows": restated,
            "min_observation_date": _plain(window[0]),
            "max_observation_date": _plain(window[1]),
            "max_data_cutoff_date": _plain(window[2]),
            "errors": errors,
            "passed": not errors,
            "registry_version": self.registry.version,
        }

    def _restated(self, path: Path, table: str, grain: list[str]) -> int:
        """Grain rows in the file that already exist: they will be restatements."""
        keys = ", ".join(quoted(column) for column in grain)
        matches = " AND ".join(
            f"CAST(s.{quoted(c)} AS VARCHAR) = CAST(t.{quoted(c)} AS VARCHAR)" for c in grain
        )
        location = str(self.path).replace("'", "''")
        with duckdb.connect() as con:
            con.execute(f"CREATE TABLE staged AS SELECT DISTINCT {keys} FROM {self._reader(path)}")
            con.execute(f"ATTACH '{location}' AS source (READ_ONLY)")
            try:
                return con.execute(
                    f"SELECT count(*) FROM staged AS s WHERE EXISTS "
                    f"(SELECT 1 FROM source.{quoted(table)} AS t WHERE {matches})"
                ).fetchone()[0]
            finally:
                con.execute("DETACH source")

    # -- append --------------------------------------------------------------------------
    def append(self, staged_id: str, table: str, file_name: str | None = None) -> dict:
        with self.locked():
            report = self.validate(staged_id, table)
            if not report["passed"]:
                raise SourceError("Validation failed; nothing appended")
            path = self.staged(staged_id)
            spec = self.table_spec(table)
            load_column = self.registry.data["point_in_time"]["load_column"]
            with duckdb.connect(str(self.path)) as con:
                con.execute("BEGIN TRANSACTION")
                try:
                    previous = con.execute(
                        "SELECT load_id, chain_hash FROM source_loads ORDER BY load_id DESC LIMIT 1"
                    ).fetchone()
                    load_id = previous[0] + 1
                    types = {row[0]: row[1] for row in con.execute(f"DESCRIBE {quoted(table)}").fetchall()}
                    columns = [column for column in spec["allowed_columns"] if column != load_column]
                    projection = ", ".join(
                        f"CAST({quoted(column)} AS {types[column]})" for column in columns
                    )
                    target = ", ".join(quoted(column) for column in [*columns, load_column])
                    con.execute(
                        f"INSERT INTO {quoted(table)} ({target}) SELECT {projection}, ? FROM {self._reader(path)}",
                        [load_id],
                    )
                    entry = self._entry(
                        con,
                        table,
                        f"(SELECT * FROM {quoted(table)} WHERE {quoted(load_column)} = {load_id})",
                        load_id,
                        file_name or path.name,
                        report["file_sha256"],
                        report["restated_rows"],
                    )
                    if entry["row_count"] != report["rows"]:
                        raise SourceError("Appended row count differs from the validated file")
                    self._write_ledger(con, load_id, [entry], previous[1])
                    con.execute("COMMIT")
                except Exception:
                    con.execute("ROLLBACK")
                    raise
        return next(item for item in self.ledger() if item["load_id"] == load_id)

    def _entry(self, con, table, relation, load_id, file_name, digest, restated):
        cutoff = quoted(self.registry.data["point_in_time"]["data_cutoff_column"])
        count, low, high, latest = con.execute(
            f"SELECT count(*), min(observation_date), max(observation_date), max({cutoff}) FROM {relation}"
        ).fetchone()
        return {
            "load_id": load_id,
            "table_name": table,
            "file_name": file_name,
            "file_sha256": digest,
            "row_count": count,
            "restated_rows": restated,
            "min_observation_date": low,
            "max_observation_date": high,
            "max_data_cutoff_date": latest,
            "registry_version": self.registry.version,
            "loaded_at": datetime.now(UTC).isoformat(),
        }

    def _write_ledger(self, con, load_id, entries, previous):
        chain_hash = chain(previous, entries)
        for entry in entries:
            con.execute(
                "INSERT INTO source_loads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    load_id,
                    entry["table_name"],
                    entry["file_name"],
                    entry["file_sha256"],
                    entry["row_count"],
                    entry["restated_rows"],
                    entry["min_observation_date"],
                    entry["max_observation_date"],
                    entry["max_data_cutoff_date"],
                    entry["registry_version"],
                    entry["loaded_at"],
                    previous,
                    chain_hash,
                ],
            )


def _plain(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _literal(value):
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"
