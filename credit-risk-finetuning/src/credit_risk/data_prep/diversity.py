"""Deterministic template-clone, family-leakage, and near-miss controls."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from credit_risk.data_prep.taxonomy import Situation, dimensions
from credit_risk.review_store import digest

MAX_PER_TEMPLATE_FAMILY = 50
NUMBER = re.compile(r"(?<![A-Za-z_])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?")


def template_family(record):
    case = record.get("case", record)
    provenance = case.get("provenance", {})
    family = record.get("template_family") or provenance.get("template_family")
    if not family:
        raise ValueError("Explicit template_family required")
    return str(family)


def clone_skeleton(record):
    case = record.get("case", record)
    question = record.get("question", case.get("question", ""))
    target = record.get("target", case.get("target"))
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except json.JSONDecodeError:
            pass
    text = json.dumps({"question": question, "target": target}, sort_keys=True, default=str)
    normalized = " ".join(NUMBER.sub("<n>", text).casefold().split())
    return digest(normalized)


def diversity_report(records, maximum=MAX_PER_TEMPLATE_FAMILY):
    family_counts = Counter()
    skeleton_counts = Counter()
    family_splits = defaultdict(set)
    family_situations = defaultdict(set)
    near_miss_required = set()
    for record in records:
        case = record.get("case", record)
        family = template_family(record)
        values = dimensions(record)
        family_counts[family] += 1
        skeleton_counts[clone_skeleton(record)] += 1
        family_splits[family].add(record.get("split", case.get("split")))
        family_situations[family].add(values["situation"])
        provenance = case.get("provenance", {})
        if record.get("requires_near_miss") or provenance.get("requires_near_miss"):
            near_miss_required.add(family)
    violations = []
    violations.extend(
        f"template_family_cap:{family}:{count}"
        for family, count in family_counts.items()
        if count > maximum
    )
    violations.extend(
        f"clone_skeleton_cap:{skeleton}:{count}"
        for skeleton, count in skeleton_counts.items()
        if count > maximum
    )
    violations.extend(
        f"template_family_split_leakage:{family}"
        for family, splits in family_splits.items()
        if len({split for split in splits if split}) > 1
    )
    violations.extend(
        f"missing_near_miss:{family}"
        for family in near_miss_required
        if Situation.NEAR_MISS.value not in family_situations[family]
    )
    return {
        "maximum_per_template_family": maximum,
        "template_family_counts": dict(sorted(family_counts.items())),
        "skeleton_counts": dict(sorted(skeleton_counts.items())),
        "situation_counts": dict(Counter(dimensions(record)["situation"] for record in records)),
        "violations": sorted(violations),
        "passed": not violations,
    }


def validate_diversity(records, maximum=MAX_PER_TEMPLATE_FAMILY):
    report = diversity_report(records, maximum)
    if report["violations"]:
        raise ValueError("Dataset diversity failed: " + ", ".join(report["violations"]))
    return report
