from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"not_started", "in_progress", "blocked", "completed"}


def test_enrichment_tracker_is_consistent():
    tracker = yaml.safe_load((ROOT / "docs/enrichment-status.yaml").read_text())
    phases = tracker["phases"]
    identities = [phase["id"] for phase in phases]
    assert identities == sorted(set(identities))
    by_id = {phase["id"]: phase for phase in phases}
    for phase in phases:
        assert phase["status"] in ALLOWED
        assert all(dependency in by_id for dependency in phase["dependencies"])
        if phase["status"] == "completed":
            assert phase["completed_at"] and phase["evidence"]
            assert all(by_id[item]["status"] == "completed" for item in phase["dependencies"])
        else:
            assert phase["completed_at"] is None
    current = by_id[tracker["current_phase"]]
    assert current["status"] != "completed"
    eligible = [
        phase["id"]
        for phase in phases
        if phase["status"] not in ("completed", "blocked")
        and all(by_id[item]["status"] == "completed" for item in phase["dependencies"])
    ]
    assert tracker["next_phase"] == min(eligible)
    verified = tracker["last_verified"]
    assert verified["tests_passed"] > 0
    assert verified["tests_skipped"] >= 0
    assert verified["ruff"] == "passed"


def test_pitfalls_baseline_matches_tracker():
    tracker = yaml.safe_load((ROOT / "docs/enrichment-status.yaml").read_text())
    verified = tracker["last_verified"]
    pitfalls = (ROOT / "docs/find-the-pitfalls-in-crystalline-teacup.md").read_text()
    expected = f"**{verified['tests_passed']} passed, {verified['tests_skipped']} skipped,"
    assert expected in pitfalls
