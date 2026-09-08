"""Append-only persistence for feedback records, including SQL reviews.

The handover repository had the FeedbackRecord schema but nothing that ever wrote one, so
a reviewer's approval or correction had nowhere to go. This is the missing sink.

Records are appended as JSON Lines. The feedback worker reads the same file, which is why
the API writes it and never reads it back for decisions.
"""
from __future__ import annotations

from pathlib import Path

from credit_risk.schemas import FeedbackRecord


class FeedbackStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: FeedbackRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def read_all(self) -> list[FeedbackRecord]:
        if not self.path.is_file():
            return []
        records: list[FeedbackRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(FeedbackRecord.model_validate_json(line))
        return records
