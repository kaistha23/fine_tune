"""100 synthetic mechanics-only examples. Not SME-approved credit training data."""

import argparse
import json
from datetime import date
from pathlib import Path

import yaml

from credit_risk.dataset import build_dataset, stable_split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    root = Path("outputs/spike-v3")
    if root.exists():
        raise ValueError("Spike directory already exists")
    payloads = []
    for n in range(100):
        portfolio = ["retail"] * 5 + ["sme"] * 3 + ["corporate"] * 2
        asof = "2026-02-01" if n >= 90 else "2025-06-30"
        case = {
            "case_id": f"SYNTH-{n}-{asof}",
            "obligor_id": f"SYNTH-{n}",
            "group_id": f"SYNTH-G-{n}",
            "portfolio": portfolio[n % 10],
            "jurisdiction": "SAMA" if n % 2 else "CBUAE",
            "as_of_date": asof,
            "current_position": {"days_past_due": n % 31, "stage": 1 + (n % 3)},
        }
        claim = f"current_position.days_past_due = {n % 31}."
        target = {
            "answer_status": "ANSWERED",
            "executive_summary": claim,
            "facts": [{"statement": claim, "evidence_ids": [case["case_id"]]}],
            "human_approval_required": True,
        }
        payloads.append(
            {
                "case": case,
                "question": (
                    "Report the supplied days past due."
                    if n < 50
                    else "State the current delinquency in days."
                ),
                "evidence": [],
                "target": json.dumps(target),
                "task_type": "factsheet",
                "situation": "base",
                "template_family": f"mechanics-{stable_split(case['group_id'])}-{n // 50}",
                "data_classification": "synthetic",
                "review": {
                    "status": "approved",
                    "reviewer_id": "synthetic-fixture-validator",
                    "quality_score": 5,
                },
            }
        )
    manifest = build_dataset(payloads, root / "dataset", date(2026, 1, 1))
    cfg = yaml.safe_load(Path("configs/training.yaml").read_text())
    cfg.update(
        data=str(root / "dataset"),
        adapter_path="adapters/candidates/spike-v3",
        spike_iters=24,
        steps_per_report=8,
        steps_per_eval=8,
        save_every=8,
        val_batches=2,
    )
    (root / "training.yaml").write_text(yaml.safe_dump(cfg))
    (root / "purpose.txt").write_text(
        "Mechanics and memory only. Synthetic deterministic targets; no domain accuracy claim.\n"
    )
    print(manifest["counts"])


if __name__ == "__main__":
    main()
