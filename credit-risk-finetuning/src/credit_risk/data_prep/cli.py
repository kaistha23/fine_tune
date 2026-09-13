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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fixture", "gold", "spike", "coverage", "diversity"))
    args, remaining = parser.parse_known_args(argv)
    if args.command == "fixture":
        from credit_risk.data_prep.fixture import main as command
    elif args.command == "gold":
        from credit_risk.data_prep.gold import main as command
    elif args.command == "spike":
        from credit_risk.data_prep.spike import main as command
    else:
        return report_command(args.command, remaining)
    command(remaining)


if __name__ == "__main__":
    main()
