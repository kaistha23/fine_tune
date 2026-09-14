"""Factsheet construction and the SFT dataset format (findings H7 and H1)."""

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

from credit_risk.dataset import build_sft_record, stable_split, to_chat_line
from credit_risk.factsheet import FactsheetError, build_factsheet
from credit_risk.schemas import CreditFactsheet, Jurisdiction, Portfolio, QueryPlan

ROOT = Path(__file__).parents[1]


def plan(**overrides) -> QueryPlan:
    base = {
        "portfolio": Portfolio.CORPORATE,
        "jurisdiction": Jurisdiction.SAMA,
        "obligor_id": "OBL-0008",
        "date_from": date(2025, 1, 31),
        "date_to": date(2025, 3, 31),
        "as_of_date": date(2025, 4, 15),
        "metrics": ["current_ratio", "utilisation_pct", "pit_pd", "stage"],
    }
    base.update(overrides)
    return QueryPlan(**base)


def rows() -> list[dict]:
    return [
        {
            "observation_date": "2025-01-31",
            "obligor_id": "OBL-0008",
            "current_assets": 120.0,
            "current_liabilities": 100.0,
            "outstanding": 500.0,
            "facility_limit": 1000.0,
            "pit_pd": 0.02,
            "lgd": 0.4,
            "ead": 520.0,
            "ecl": 4.16,
            "stage": 1,
            "days_past_due": 0,
            "internal_rating": "3",
        },
        {
            "observation_date": "2025-02-28",
            "obligor_id": "OBL-0008",
            "current_assets": 110.0,
            "current_liabilities": 100.0,
            "outstanding": 650.0,
            "facility_limit": 1000.0,
            "pit_pd": 0.035,
            "lgd": 0.4,
            "ead": 670.0,
            "ecl": 9.38,
            "stage": 2,
            "days_past_due": 35,
            "internal_rating": "5",
        },
        {
            "observation_date": "2025-03-31",
            "obligor_id": "OBL-0008",
            "current_assets": 90.0,
            "current_liabilities": 100.0,
            "outstanding": 800.0,
            "facility_limit": 1000.0,
            "pit_pd": 0.05,
            "lgd": 0.4,
            "ead": 820.0,
            "ecl": 16.4,
            "stage": 2,
            "days_past_due": 41,
            "internal_rating": "6",
        },
    ]


class FactsheetTests(unittest.TestCase):
    def test_metrics_are_computed_deterministically_from_the_latest_row(self) -> None:
        sheet = build_factsheet(rows(), plan())
        self.assertIsInstance(sheet, CreditFactsheet)
        self.assertEqual(sheet.observation_months, 3)
        self.assertEqual(sheet.calculated_metrics["current_ratio"].value, 0.9)
        self.assertEqual(sheet.calculated_metrics["current_ratio"].unit, "x")
        self.assertEqual(
            sheet.calculated_metrics["current_ratio"].formula_id, "ratio.current_ratio.v2"
        )
        self.assertEqual(sheet.calculated_metrics["utilisation_pct"].value, 80.0)
        self.assertEqual(sheet.calculated_metrics["utilisation_pct"].unit, "pct")

    def test_probability_change_is_reported_in_percentage_points(self) -> None:
        # 2% to 5% is +3pp. Reporting +150% here is the classic numeric error both
        # plans single out.
        sheet = build_factsheet(rows(), plan())
        self.assertEqual(sheet.trends["pit_pd_change_pp"], 3.0)
        self.assertEqual(sheet.trends["utilisation_change_pp"], 30.0)

    def test_model_outputs_are_reported_never_recomputed(self) -> None:
        sheet = build_factsheet(rows(), plan())
        self.assertEqual(sheet.model_outputs["pit_pd"], 0.05)
        self.assertEqual(sheet.model_outputs["ecl"], 16.4)
        # ECL is not in calculations.CALCULATORS, so nothing derived it.
        self.assertNotIn("ecl", {k for k, v in sheet.calculated_metrics.items() if v.formula_id})

    def test_events_and_migration_are_detected(self) -> None:
        sheet = build_factsheet(rows(), plan())
        self.assertIn("stage_2_observed_in_window", sheet.events)
        self.assertIn("stage_migration_1_to_2", sheet.events)
        self.assertIn("2_months_at_30dpd_or_worse", sheet.events)

    def test_missing_inputs_are_surfaced_not_silently_dropped(self) -> None:
        incomplete = rows()
        incomplete[-1]["current_assets"] = None
        sheet = build_factsheet(incomplete, plan())
        self.assertTrue(sheet.calculated_metrics["current_ratio"].missing_data_flag)
        self.assertEqual(sheet.calculated_metrics["current_ratio"].validation_status, "warning")
        self.assertIn("current_assets", sheet.missing_information)

    def test_stale_data_is_flagged(self) -> None:
        sheet = build_factsheet(
            rows(), plan(as_of_date=date(2026, 1, 15), date_to=date(2025, 3, 31))
        )
        self.assertTrue(any("months_old" in flag for flag in sheet.data_quality_flags))

    def test_empty_result_is_refused(self) -> None:
        with self.assertRaises(FactsheetError):
            build_factsheet([], plan())


class DatasetFormatTests(unittest.TestCase):
    def test_training_line_is_an_object_keyed_by_messages(self) -> None:
        record = build_sft_record(
            {
                "obligor_id": "OBL-0008",
                "portfolio": "corporate",
                "jurisdiction": "SAMA",
                "group_id": "OBL-0008",
                "as_of_date": "2025-12-31",
            },
            json.dumps({"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}),
            "credit_deterioration",
        )
        parsed = json.loads(to_chat_line(record))
        # The original wrote a bare array here, which mlx-lm's chat loader rejects.
        self.assertIsInstance(parsed, dict)
        self.assertIn("messages", parsed)
        self.assertEqual([m["role"] for m in parsed["messages"]], ["system", "user", "assistant"])

    def test_split_is_stable_per_obligor(self) -> None:
        self.assertEqual(stable_split("OBL-0008"), stable_split("OBL-0008"))
        self.assertIn(stable_split("OBL-0008"), {"train", "valid", "test"})

    def test_provenance_survives_the_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cases.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "case": {
                            "obligor_id": "OBL-0008",
                            "portfolio": "corporate",
                            "jurisdiction": "SAMA",
                            "group_id": "OBL-0008",
                            "as_of_date": "2025-12-31",
                        },
                        "target": json.dumps(
                            {"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""}
                        ),
                        "task_type": "credit_deterioration",
                        "situation": "base",
                        "template_family": "factsheet-round-trip",
                        "review": {
                            "status": "approved",
                            "reviewer_id": "synthetic",
                            "quality_score": 5,
                        },
                        "data_classification": "synthetic",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out = Path(tmp) / "out"
            (Path(tmp) / "exclusions.json").write_text('{"groups":[],"content_hashes":[]}')
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "credit_risk.dataset",
                    str(source),
                    str(out),
                    "--out-of-time-from",
                    "2027-01-01",
                    "--exclusions",
                    str(Path(tmp) / "exclusions.json"),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            written = [p for p in out.glob("*.jsonl") if not p.name.endswith(".provenance.jsonl")]
            lines = [line for p in written for line in p.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 1)
            self.assertIn("messages", json.loads(lines[0]))

            prov = [
                line
                for p in out.glob("*.provenance.jsonl")
                for line in p.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(prov), 1)
            meta = json.loads(prov[0])
            for key in (
                "example_id",
                "portfolio",
                "task_type",
                "jurisdiction",
                "split",
                "dataset_version",
            ):
                self.assertIn(key, meta)


if __name__ == "__main__":
    unittest.main()
