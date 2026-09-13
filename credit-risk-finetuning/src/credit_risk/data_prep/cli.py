"""Unified data-preparation command line."""

from __future__ import annotations

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fixture", "gold", "spike"))
    args, remaining = parser.parse_known_args(argv)
    if args.command == "fixture":
        from credit_risk.data_prep.fixture import main as command
    elif args.command == "gold":
        from credit_risk.data_prep.gold import main as command
    else:
        from credit_risk.data_prep.spike import main as command
    command(remaining)


if __name__ == "__main__":
    main()
