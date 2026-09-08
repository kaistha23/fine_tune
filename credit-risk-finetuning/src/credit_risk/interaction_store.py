"""Append-only persistence for served interactions, so a correction has something to correct.

The feedback loop had no input. TRAINING_ROOT_CAUSES is {"model_behaviour"}, but nothing
could create a record with that root cause: /v1/query/review only produces schema, query,
calculation and data causes, and /v1/analyse returned no handle to attach a correction to.
Every correction an analyst might have written was unrecordable.

This is the missing half. /v1/analyse writes what it served here; /v1/analyse/feedback
reads it back to rebuild the exact input the model saw.

Reads are by interaction_id and are a linear scan. That is fine for a JSON Lines store at
review volumes - one lookup per submitted correction, not per query - and keeping the
format the same as the feedback log means the two can be inspected with the same tools.
"""
from __future__ import annotations

from pathlib import Path

from credit_risk.schemas import InteractionRecord


class InteractionNotFound(LookupError):
    """Raised when a correction names an interaction that was never served.

    Distinguished from a malformed request because the two mean different things: a
    missing interaction is usually a rotated store, not a bad submission.
    """


class InteractionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: InteractionRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def get(self, interaction_id: str) -> InteractionRecord:
        """The last record written for this id.

        Last rather than first: a replayed interaction should be corrected against what
        was most recently served, not against a stale earlier answer.
        """
        found: InteractionRecord | None = None
        if not self.path.is_file():
            raise InteractionNotFound(interaction_id)
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = InteractionRecord.model_validate_json(line)
                if record.interaction_id == interaction_id:
                    found = record
        if found is None:
            raise InteractionNotFound(interaction_id)
        return found
