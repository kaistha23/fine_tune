import json
from pathlib import Path

import duckdb
import pytest

from credit_risk.data_prep.cli import main
from credit_risk.evaluation.gold import content_hash


ROOT = Path(__file__).resolve().parents[1]


def test_unified_fixture_command_builds_the_governed_grains(tmp_path):
    output = tmp_path / "fixture.duckdb"
    main(["fixture", "--out", str(output), "--obligors", "4", "--seed", "17"])
    with duckdb.connect(str(output), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM obligor_monthly").fetchone()[0] == 48
        assert con.execute("SELECT count(*) FROM facility_monthly").fetchone()[0] > 0
        assert (
            con.execute(
                "SELECT count(*) FROM (SELECT obligor_id, observation_date "
                "FROM obligor_monthly GROUP BY ALL HAVING count(*) > 1)"
            ).fetchone()[0]
            == 0
        )


def test_unified_gold_command_seals_output_and_old_scripts_are_removed(tmp_path):
    output = tmp_path / "gold.jsonl"
    main(["gold", "--out", str(output)])
    rows = [json.loads(line) for line in output.read_text().splitlines() if not line.startswith("#")]
    manifest = rows[-1]
    assert manifest["case_count"] == 8
    assert manifest["content_hash"] == content_hash(output)
    assert "credit-risk-data-prep gold" in output.read_text()
    for name in ("make_fixture.py", "make_gold_set.py", "make_training_spike.py"):
        assert not (ROOT / "scripts" / name).exists()


def test_unified_coverage_and_diversity_commands_write_machine_readable_reports(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "task_type": "ews",
                "situation": "base",
                "portfolio": "corporate",
                "jurisdiction": "SAMA",
                "template_family": "ews-corporate-sama-base",
                "split": "train",
                "question": "Assess obligor 42",
                "target": {"answer": "Watch utilisation at 82%."},
            }
        )
        + "\n"
    )
    targets = tmp_path / "targets.yaml"
    targets.write_text(
        "minimum_per_cell: 1\nrequired_cells:\n"
        "  - {task_type: ews_analysis, portfolio: corporate, jurisdiction: SAMA, situation: base}\n"
    )
    coverage = tmp_path / "coverage.json"
    diversity = tmp_path / "diversity.json"

    main(["coverage", str(records), "--targets", str(targets), "--out", str(coverage)])
    main(["diversity", str(records), "--out", str(diversity)])

    assert json.loads(coverage.read_text())["passed"] is True
    assert json.loads(diversity.read_text())["passed"] is True


def test_unified_coverage_command_fails_when_required_cells_are_missing(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text("")
    targets = tmp_path / "targets.yaml"
    targets.write_text(
        "minimum_per_cell: 1\nrequired_cells:\n"
        "  - {task_type: policy_qa, portfolio: sme, jurisdiction: SAMA, situation: base}\n"
    )
    with pytest.raises(SystemExit):
        main(["coverage", str(records), "--targets", str(targets)])
