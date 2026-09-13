"""Measure required task × portfolio × jurisdiction × situation cells."""

from __future__ import annotations

from collections import Counter

from credit_risk.data_prep.taxonomy import Situation, dimensions, normalize_task_type


KEYS = ("task_type", "portfolio", "jurisdiction", "situation")


def coverage_key(values):
    return "|".join(values[key] for key in KEYS)


def coverage_report(records, targets):
    counts = Counter(coverage_key(dimensions(record)) for record in records)
    default = int(targets.get("minimum_per_cell", 30))
    required = []
    for cell in targets.get("required_cells", []):
        unknown = set(cell) - set(KEYS) - {"minimum"}
        if unknown or any(key not in cell for key in KEYS):
            raise ValueError("Coverage cells must declare exactly the four taxonomy dimensions")
        values = {key: str(cell[key]) for key in KEYS}
        values["task_type"] = normalize_task_type(values["task_type"])
        values["situation"] = Situation(values["situation"]).value
        key = coverage_key(values)
        minimum = int(cell.get("minimum", default))
        if minimum <= 0:
            raise ValueError("Coverage minimum must be positive")
        required.append({**values, "minimum": minimum, "actual": counts.get(key, 0)})
    missing = [cell for cell in required if cell["actual"] < cell["minimum"]]
    return {
        "dimensions": list(KEYS),
        "counts": dict(sorted(counts.items())),
        "required_cells": required,
        "missing_cells": missing,
        "passed": not missing,
    }
