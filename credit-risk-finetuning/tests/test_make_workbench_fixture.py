import subprocess
import sys
from pathlib import Path

from credit_risk.workbench.contracts import Case, default_version, inspect_dataset
from credit_risk.workbench.evaluation import assess_training_target

ROOT = Path(__file__).parents[1]


def test_generated_local_dataset_is_registrable_and_targets_are_admissible(tmp_path):
    source = tmp_path / "credit_risk.duckdb"
    output = tmp_path / "workbench-v1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "credit_risk.data_prep.fixture",
            "--out",
            str(source),
            "--obligors",
            "12",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/make_workbench_fixture.py"),
            "--source",
            str(source),
            "--out",
            str(output),
            "--version",
            "test.1",
            "--train",
            "4",
            "--validation",
            "2",
            "--test",
            "2",
            "--oot",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    inspected = inspect_dataset(output / "manifest.json")
    assert inspected["counts"] == {"train": 4, "validation": 2, "test": 2, "oot": 2}
    assert inspected["diversity"]["passed"] is True
    assert max(inspected["diversity"]["template_family_counts"].values()) <= 2
    version = default_version("credit_analysis")
    for raw in inspected["cases"]:
        case = Case.model_validate(raw)
        assert case.fact_records[0].unit == "stage"
        if case.target is not None:
            _, failures, parsed = assess_training_target(case, case.target, version)
            assert parsed and not failures
