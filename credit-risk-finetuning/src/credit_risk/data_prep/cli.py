"""Unified data-preparation command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def records(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def report_command(kind, argv):
    parser = argparse.ArgumentParser(prog=f"credit-risk-data-prep {kind}")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    if kind == "coverage":
        parser.add_argument(
            "--targets", type=Path, default=Path("configs/data_prep/coverage_targets.yaml")
        )
    args = parser.parse_args(argv)
    if kind == "coverage":
        from credit_risk.data_prep.coverage import coverage_report

        report = coverage_report(records(args.input), yaml.safe_load(args.targets.read_text()))
    else:
        from credit_risk.data_prep.diversity import diversity_report

        report = diversity_report(records(args.input))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


def rules_command(argv):
    parser = argparse.ArgumentParser(prog="credit-risk-data-prep rules")
    parser.add_argument("input", type=Path, help="JSON containing factsheet and evidence")
    parser.add_argument("--registry", type=Path, default=Path("configs/policy_rules.yaml"))
    parser.add_argument(
        "--schema-registry", type=Path, default=Path("configs/schema_registry.yaml")
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    from credit_risk.data_prep.rules import PolicyRuleRegistry, evaluate_rules
    from credit_risk.query_guard import SchemaRegistry
    from credit_risk.schemas import CreditFactsheet, Evidence

    payload = json.loads(args.input.read_text())
    registry = PolicyRuleRegistry(
        args.registry, schema_registry=SchemaRegistry(args.schema_registry)
    )
    evaluations = evaluate_rules(
        registry,
        CreditFactsheet.model_validate(payload["factsheet"]),
        [Evidence.model_validate(item) for item in payload.get("evidence", [])],
    )
    report = {
        "registry_version": registry.version,
        "evaluations": [item.model_dump(mode="json") for item in evaluations],
        "mandatory_unevaluable": [
            item.rule_id
            for item in evaluations
            if item.mandatory and item.status == "unevaluable"
        ],
    }
    report["passed"] = not report["mandatory_unevaluable"]
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


def source_command(kind, argv):
    parser = argparse.ArgumentParser(prog=f"credit-risk-data-prep {kind}")
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/workbench/source"),
        help="Workbench source directory holding credit_risk.duckdb and its ledger",
    )
    parser.add_argument(
        "--schema-registry", type=Path, default=Path("configs/schema_registry.yaml")
    )
    if kind == "source-init":
        parser.add_argument("--from", dest="source", type=Path, default=Path("data/curated/credit_risk.duckdb"))
    else:
        parser.add_argument("--table", required=True)
        parser.add_argument("--file", type=Path, required=True)
        parser.add_argument("--dry-run", action="store_true", help="Validate only; append nothing")
    args = parser.parse_args(argv)
    from credit_risk.query_guard import SchemaRegistry
    from credit_risk.settings import settings
    from credit_risk.workbench.sources import SourceDatabase

    database = SourceDatabase(
        args.root, SchemaRegistry(args.schema_registry, settings.schema_registry_version)
    )
    if kind == "source-init":
        result = database.initialize(args.source)
    else:
        staged = database.stage(args.file.read_bytes(), args.file.name)
        result = database.validate(staged["staged_id"], args.table)
        if result["passed"] and not args.dry_run:
            result = {"validation": result, "appended": database.append(
                staged["staged_id"], args.table, args.file.name
            )}
    print(json.dumps(result, indent=2, default=str))
    if kind == "load" and not (result.get("passed") or result.get("appended")):
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("fixture", "gold", "spike", "coverage", "diversity", "rules", "source-init", "load"),
    )
    args, remaining = parser.parse_known_args(argv)
    if args.command == "fixture":
        from credit_risk.data_prep.fixture import main as command
    elif args.command == "gold":
        from credit_risk.data_prep.gold import main as command
    elif args.command == "spike":
        from credit_risk.data_prep.spike import main as command
    elif args.command == "rules":
        return rules_command(remaining)
    elif args.command in ("source-init", "load"):
        return source_command(args.command, remaining)
    else:
        return report_command(args.command, remaining)
    command(remaining)


if __name__ == "__main__":
    main()
