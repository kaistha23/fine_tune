import json
from pathlib import Path

import duckdb

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
