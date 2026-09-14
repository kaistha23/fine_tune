from datetime import date
from pathlib import Path

import duckdb
import pytest

from credit_risk.data_prep.fixture import export_month
from credit_risk.data_prep.fixture import main as fixture
from credit_risk.data_service import _execute_with_limits, validate_result
from credit_risk.query_guard import GuardedQueryCompiler, SchemaRegistry
from credit_risk.schemas import QueryPlan
from credit_risk.workbench.sources import SourceDatabase, SourceError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def registry():
    return SchemaRegistry(ROOT / "configs/schema_registry.yaml")


@pytest.fixture
def source(tmp_path, registry):
    curated = tmp_path / "curated.duckdb"
    fixture(["--out", str(curated), "--obligors", "4", "--seed", "17"])
    database = SourceDatabase(tmp_path / "source", registry)
    database.initialize(curated)
    return database


def stage_file(database, path):
    return database.stage(Path(path).read_bytes(), Path(path).name)["staged_id"]


def obligor_rows(database, table="obligor_monthly", where="obligor_id = 'OBL-0001'"):
    with duckdb.connect(str(database.path), read_only=True) as con:
        cursor = con.execute(f"SELECT * FROM {table} WHERE {where}")
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def write_csv(path, rows):
    columns = list(rows[0])
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join("" if row[c] is None else str(row[c]) for c in columns))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_initialize_records_load_one_and_never_overwrites(source, tmp_path):
    summary = source.summary()
    assert summary["initialized"] and summary["snapshot"]["load_id"] == 1
    assert {entry["table_name"] for entry in summary["ledger"]} == {
        "obligor_monthly",
        "facility_monthly",
    }
    assert summary["tables"]["obligor_monthly"] == 48
    with pytest.raises(SourceError, match="already initialized"):
        source.initialize(tmp_path / "curated.duckdb")


def test_month_append_extends_ledger_chain(source, tmp_path):
    files = export_month(17, 4, "2026-01", tmp_path / "incoming")
    first = source.snapshot()
    report = source.validate(stage_file(source, files["obligor_monthly"]), "obligor_monthly")
    assert report["passed"], report["errors"]
    assert report["rows"] == 4 and report["restated_rows"] == 0
    entry = source.append(report["staged_id"], "obligor_monthly", "obligor-2026-01.parquet")
    assert entry["load_id"] == 2 and entry["prev_chain_hash"] == first["chain_hash"]
    assert source.snapshot()["chain_hash"] != first["chain_hash"]
    assert source.snapshot(1) == first
    assert source.summary()["tables"]["obligor_monthly"] == 52
    again = source.validate(stage_file(source, files["obligor_monthly"]), "obligor_monthly")
    assert {error["code"] for error in again["errors"]} == {"file_already_loaded"}
    with pytest.raises(SourceError, match="Validation failed"):
        source.append(again["staged_id"], "obligor_monthly")


def test_validation_rejects_contract_violations(source, tmp_path):
    base = obligor_rows(source)[0]
    base.pop("load_id")
    base.update(observation_date=date(2026, 2, 28), data_cutoff_date=date(2026, 3, 5), model_run_date=date(2026, 3, 10))
    bad = [
        {**base, "stage": 7},
        {**base, "observation_date": date(2026, 3, 31), "data_cutoff_date": date(2026, 3, 1)},
        {**base, "observation_date": date(2026, 4, 30), "jurisdiction": "FCA"},
        {**base, "observation_date": date(2026, 5, 31), "pit_pd": 1.5},
        {**base, "observation_date": date(2026, 6, 30), "days_past_due": "late"},
        {**base, "observation_date": date(2026, 7, 31), "obligor_id": None},
        {**base, "observation_date": date(2026, 7, 31), "data_cutoff_date": date(2099, 1, 1), "model_run_date": date(2099, 1, 2)},
        dict(base),
        dict(base),
    ]
    report = source.validate(stage_file(source, write_csv(tmp_path / "bad.csv", bad)), "obligor_monthly")
    codes = {error["code"] for error in report["errors"]}
    assert not report["passed"]
    assert "invalid_type:days_past_due" in codes
    report = source.validate(
        stage_file(source, write_csv(tmp_path / "typed.csv", [row for row in bad if row["days_past_due"] != "late"])),
        "obligor_monthly",
    )
    codes = {error["code"] for error in report["errors"]}
    assert {
        "not_allowed:stage",
        "cutoff_before_observation",
        "not_allowed:jurisdiction",
        "above_maximum:pit_pd",
        "missing_required:obligor_id",
        "future_data_cutoff",
        "duplicate_grain",
    } <= codes
    extra = [{**base, "load_id": 9}]
    report = source.validate(stage_file(source, write_csv(tmp_path / "extra.csv", extra)), "obligor_monthly")
    assert {"code": "undeclared_columns", "detail": ["load_id"]} in report["errors"]
    missing = [{k: v for k, v in base.items() if k != "stage"}]
    report = source.validate(stage_file(source, write_csv(tmp_path / "missing.csv", missing)), "obligor_monthly")
    assert {"code": "missing_columns", "detail": ["stage"]} in report["errors"]
    with pytest.raises(SourceError):
        source.validate(stage_file(source, tmp_path / "missing.csv"), "unregistered_table")
    with pytest.raises(SourceError, match="Parquet, CSV or JSONL"):
        source.stage(b"x", "rows.xlsx")


def plan(row, as_of):
    return QueryPlan(
        portfolio=row["portfolio"],
        jurisdiction=row["jurisdiction"],
        obligor_id=row["obligor_id"],
        date_from=date(2025, 1, 1),
        date_to=min(as_of, date(2025, 12, 31)),
        as_of_date=as_of,
        metrics=["stage", "pit_pd"],
        analysis_type="factsheet",
    )


def run(registry, source, query_plan, watermark):
    compiled = GuardedQueryCompiler(registry).compile(query_plan, snapshot_load_id=watermark)
    rows = _execute_with_limits(compiled, registry.data["query_controls"], database_path=source.path)
    validate_result(rows, compiled, query_plan, registry.data["query_controls"])
    return compiled, rows


def test_restatement_is_visible_only_to_later_snapshots_and_known_dates(source, registry, tmp_path):
    original = obligor_rows(source, where="obligor_id = 'OBL-0001' AND observation_date = DATE '2025-06-30'")[0]
    restated = {k: v for k, v in original.items() if k != "load_id"}
    restated.update(stage=3 if original["stage"] != 3 else 1, data_cutoff_date=date(2026, 2, 1), model_run_date=date(2026, 2, 2))
    report = source.validate(stage_file(source, write_csv(tmp_path / "restate.csv", [restated])), "obligor_monthly")
    assert report["passed"], report["errors"]
    assert report["restated_rows"] == 1
    source.append(report["staged_id"], "obligor_monthly")

    def june(rows):
        return next(row["stage"] for row in rows if row["observation_date"] == "2025-06-30")

    late = plan(original, date(2026, 3, 1))
    _compiled, at_one = run(registry, source, late, 1)
    compiled, at_two = run(registry, source, late, 2)
    assert june(at_one) == original["stage"]
    assert june(at_two) == restated["stage"]
    assert len(at_two) == len(at_one)
    assert compiled.parameters[0] == 2 and "QUALIFY ROW_NUMBER()" in compiled.sql
    _compiled, before_known = run(registry, source, plan(original, date(2026, 1, 15)), 2)
    assert june(before_known) == original["stage"]


def test_unsnapshotted_sql_is_unchanged_and_cohorts_resolve_snapshots(source, registry):
    original = obligor_rows(source)[0]
    compiler = GuardedQueryCompiler(registry)
    plain = compiler.compile(plan(original, date(2025, 12, 31)))
    assert "load_id" not in plain.sql and plain.snapshot_load_id is None
    with pytest.raises(ValueError, match="positive integer"):
        compiler.compile(plan(original, date(2025, 12, 31)), snapshot_load_id=0)
    cohort = QueryPlan(
        portfolio=original["portfolio"],
        jurisdiction=original["jurisdiction"],
        entity_level="portfolio",
        group_by=["stage"],
        cohort_aggregation="count",
        date_from=date(2025, 1, 1),
        date_to=date(2025, 12, 31),
        as_of_date=date(2025, 12, 31),
        metrics=["days_past_due"],
        analysis_type="factsheet",
    )
    compiled = compiler.compile(cohort, snapshot_load_id=1)
    assert compiled.sql.index('FROM (SELECT') < compiled.sql.index("GROUP BY")
    rows = _execute_with_limits(compiled, registry.data["query_controls"], database_path=source.path)
    assert isinstance(rows, list)


def test_workbench_source_endpoints_stage_validate_and_append(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from credit_risk.workbench import server

    monkeypatch.setattr(server, "PROJECT", tmp_path)
    curated = tmp_path / "data/curated/credit_risk.duckdb"
    curated.parent.mkdir(parents=True)
    fixture(["--out", str(curated), "--obligors", "4", "--seed", "17"])
    client = TestClient(server.create_app(tmp_path / "workspace", False))
    client.headers["X-Workbench-Token"] = client.get("/api/session").json()["token"]
    assert client.get("/api/sources").json()["initialized"] is False
    blocked = client.post(
        "/api/sources/stage?table=obligor_monthly&filename=x.csv", content=b"a\n1\n"
    ).json()
    assert {"code": "source_not_initialized"} in blocked["errors"]
    assert client.post("/api/sources/initialize", json={}).status_code == 200
    files = export_month(17, 4, "2026-01", tmp_path / "incoming")
    report = client.post(
        "/api/sources/stage?table=obligor_monthly&filename=jan.parquet",
        content=files["obligor_monthly"].read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    ).json()
    assert report["passed"] and report["file_name"] == "jan.parquet"
    appended = client.post(
        "/api/sources/append",
        json={"staged_id": report["staged_id"], "table": "obligor_monthly", "file_name": "jan.parquet"},
    )
    assert appended.status_code == 200 and appended.json()["load_id"] == 2
    summary = client.get("/api/sources").json()
    assert summary["snapshot"]["load_id"] == 2 and summary["tables"]["obligor_monthly"] == 52
    unsafe = client.post("/api/sources/append", json={"staged_id": "../credit_risk.duckdb", "table": "obligor_monthly"})
    assert unsafe.status_code == 422
    no_token = TestClient(server.create_app(tmp_path / "other", False))
    assert no_token.post("/api/sources/initialize", json={}).status_code == 403


def test_cli_source_init_and_load(tmp_path, capsys):
    from credit_risk.data_prep.cli import main

    curated = tmp_path / "curated.duckdb"
    fixture(["--out", str(curated), "--obligors", "4", "--seed", "17"])
    root = tmp_path / "source"
    main(["source-init", "--root", str(root), "--from", str(curated)])
    files = export_month(17, 4, "2026-01", tmp_path / "incoming")
    main(["load", "--root", str(root), "--table", "facility_monthly", "--file", str(files["facility_monthly"]), "--dry-run"])
    assert '"passed": true' in capsys.readouterr().out
    main(["load", "--root", str(root), "--table", "facility_monthly", "--file", str(files["facility_monthly"])])
    assert '"load_id": 2' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["load", "--root", str(root), "--table", "facility_monthly", "--file", str(files["facility_monthly"])])


def test_synthetic_month_keeps_obligor_facilities_and_limits(tmp_path):
    from credit_risk.data_prep.fixture import build

    obligors, facilities = build(17, 12)
    files = export_month(17, 12, "2026-02", tmp_path)
    with duckdb.connect() as con:
        new_facilities = con.execute(
            f"SELECT obligor_id, facility_id FROM read_parquet('{files['facility_monthly']}')"
        ).fetchall()
        new_limits = dict(
            con.execute(
                f"SELECT obligor_id, facility_limit FROM read_parquet('{files['obligor_monthly']}')"
            ).fetchall()
        )
        month = con.execute(
            f"SELECT DISTINCT observation_date FROM read_parquet('{files['obligor_monthly']}')"
        ).fetchall()
    december = {(row[0], row[1]) for row in facilities if row[2] == date(2025, 12, 31)}
    assert set(new_facilities) == december
    assert new_limits == {row[0]: row[15] for row in obligors if row[1] == date(2025, 12, 31)}
    assert month == [(date(2026, 2, 28),)]
    with pytest.raises(ValueError, match="after the initial"):
        export_month(17, 12, "2025-06", tmp_path / "early")
