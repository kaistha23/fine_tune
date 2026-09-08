#!/usr/bin/env python3
"""Build a small synthetic DuckDB fixture matching configs/schema_registry.yaml exactly.

Tokenised identifiers only. Deterministic under --seed. Nothing here is real data.

Point-in-time columns are deliberately lagged behind observation_date: month-end data
lands 5 days later and the risk models run 10 days later. That makes the leakage control
observable - a query as at 2025-12-31 sees 11 months, not 12, because December's data was
not yet known.

    uv run python scripts/make_fixture.py
"""
from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import duckdb

# Portfolio mix from the Feedback Plan: Retail 50 / SME 30 / Corporate 20.
PORTFOLIO_MIX = ["retail"] * 5 + ["sme"] * 3 + ["corporate"] * 2
JURISDICTIONS = ["SAMA", "CBUAE"]
RATINGS = [str(n) for n in range(1, 11)]

DATA_LAG_DAYS = 5
MODEL_LAG_DAYS = 10

PROFILE = {
    "sme": dict(revenue=(18_000_000, 6_000_000), margin=(0.11, 0.05),
                leverage=(3.4, 1.2), cur=(1.25, 0.35)),
    "corporate": dict(revenue=(240_000_000, 90_000_000), margin=(0.19, 0.07),
                      leverage=(3.0, 1.3), cur=(1.45, 0.40)),
}

MONTH_ENDS = [
    date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 31), date(2025, 4, 30),
    date(2025, 5, 31), date(2025, 6, 30), date(2025, 7, 31), date(2025, 8, 31),
    date(2025, 9, 30), date(2025, 10, 31), date(2025, 11, 30), date(2025, 12, 31),
]

OBLIGOR_DDL = """
CREATE TABLE obligor_monthly (
    obligor_id VARCHAR, observation_date DATE,
    data_cutoff_date DATE, model_run_date DATE,
    portfolio VARCHAR, jurisdiction VARCHAR,
    revenue DOUBLE, current_assets DOUBLE, current_liabilities DOUBLE, ebitda DOUBLE,
    total_debt DOUBLE, cash DOUBLE, debt_service DOUBLE, operating_cash_flow DOUBLE,
    interest_expense DOUBLE, facility_limit DOUBLE, outstanding DOUBLE,
    days_past_due INTEGER, internal_rating VARCHAR,
    ttc_pd DOUBLE, pit_pd DOUBLE, lgd DOUBLE, ead DOUBLE, ecl DOUBLE,
    stage INTEGER, watchlist_flag BOOLEAN, restructuring_flag BOOLEAN
)
"""

FACILITY_DDL = """
CREATE TABLE facility_monthly (
    obligor_id VARCHAR, facility_id VARCHAR, observation_date DATE,
    data_cutoff_date DATE, model_run_date DATE,
    portfolio VARCHAR, jurisdiction VARCHAR,
    facility_limit DOUBLE, outstanding DOUBLE, undrawn_amount DOUBLE,
    days_past_due INTEGER, collateral_value DOUBLE, stage INTEGER
)
"""


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build(seed: int, n_obligors: int):
    rng = random.Random(seed)
    obligor_rows, facility_rows = [], []

    for i in range(1, n_obligors + 1):
        obligor_id = f"OBL-{i:04d}"
        portfolio = PORTFOLIO_MIX[i % len(PORTFOLIO_MIX)]
        jurisdiction = JURISDICTIONS[i % 2]
        # A fraction of obligors deteriorate across the year so the EWS and trend paths
        # have something to find.
        drift = rng.choice([0.0, 0.0, 0.0, 0.04, 0.09])
        limit = round(rng.uniform(500_000, 40_000_000), 2)
        facility_ids = [f"FAC-{i:04d}-{k}" for k in range(1, rng.randint(1, 3) + 1)]

        for month, obs in enumerate(MONTH_ENDS):
            cutoff = obs + timedelta(days=DATA_LAG_DAYS)
            model_run = obs + timedelta(days=MODEL_LAG_DAYS)
            stress = drift * month
            util = clamp(rng.gauss(0.62, 0.15) + stress, 0.0, 1.0)
            outstanding = round(limit * util, 2)
            dpd = 0 if rng.random() > 0.10 + stress else rng.choice([1, 7, 31, 62, 95])
            stage = 3 if dpd >= 90 else (2 if dpd >= 30 or stress > 0.25 else 1)
            pit_pd = clamp(rng.gauss(0.03, 0.015) + stress * 0.9, 0.0005, 0.9999)
            ttc_pd = clamp(pit_pd * rng.uniform(0.5, 0.9), 0.0005, 0.9999)
            lgd = clamp(rng.gauss(0.45, 0.10), 0.01, 0.99)
            ead = round(outstanding * rng.uniform(1.0, 1.12), 2)
            ecl = round(pit_pd * lgd * ead, 2)

            if portfolio == "retail":
                # Corporate/SME financials do not exist at retail grain. NULL is the honest
                # value and exercises MetricValue.missing_data_flag downstream.
                fin = dict.fromkeys(
                    ["revenue", "current_assets", "current_liabilities", "ebitda",
                     "total_debt", "cash", "debt_service", "operating_cash_flow",
                     "interest_expense"], None)
            else:
                p = PROFILE[portfolio]
                revenue = max(0.0, rng.gauss(*p["revenue"]) * (1 - stress))
                ebitda = revenue * clamp(rng.gauss(*p["margin"]) - stress * 0.4, -0.05, 0.6)
                total_debt = max(0.0, ebitda * clamp(rng.gauss(*p["leverage"]) + stress * 3, 0.2, 12))
                cur_liab = max(1.0, revenue * rng.uniform(0.10, 0.25))
                fin = dict(
                    revenue=round(revenue, 2),
                    current_liabilities=round(cur_liab, 2),
                    current_assets=round(cur_liab * clamp(rng.gauss(*p["cur"]) - stress, 0.25, 4.0), 2),
                    ebitda=round(ebitda, 2),
                    total_debt=round(total_debt, 2),
                    cash=round(max(0.0, total_debt * rng.uniform(0.02, 0.30)), 2),
                    debt_service=round(max(1.0, total_debt * rng.uniform(0.10, 0.30)), 2),
                    operating_cash_flow=round(ebitda * rng.uniform(0.55, 1.05), 2),
                    interest_expense=round(max(1.0, total_debt * rng.uniform(0.04, 0.09)), 2),
                )

            obligor_rows.append((
                obligor_id, obs, cutoff, model_run, portfolio, jurisdiction,
                fin["revenue"], fin["current_assets"], fin["current_liabilities"],
                fin["ebitda"], fin["total_debt"], fin["cash"], fin["debt_service"],
                fin["operating_cash_flow"], fin["interest_expense"],
                limit, outstanding, dpd,
                RATINGS[min(len(RATINGS) - 1, int(pit_pd * 40))],
                round(ttc_pd, 6), round(pit_pd, 6), round(lgd, 6), ead, ecl,
                stage, stage >= 2, bool(stress > 0.30),
            ))

            share = 1.0 / len(facility_ids)
            for facility_id in facility_ids:
                f_limit = round(limit * share, 2)
                f_out = round(outstanding * share, 2)
                facility_rows.append((
                    obligor_id, facility_id, obs, cutoff, model_run,
                    portfolio, jurisdiction,
                    f_limit, f_out, round(max(0.0, f_limit - f_out), 2),
                    dpd, round(f_out * rng.uniform(0.3, 1.4), 2), stage,
                ))

    return obligor_rows, facility_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/curated/credit_risk.duckdb"))
    parser.add_argument("--obligors", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    obligor_rows, facility_rows = build(args.seed, args.obligors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    con = duckdb.connect(str(args.out))
    con.execute(OBLIGOR_DDL)
    con.execute(FACILITY_DDL)
    con.executemany(
        "INSERT INTO obligor_monthly VALUES (" + ",".join("?" * 27) + ")", obligor_rows)
    con.executemany(
        "INSERT INTO facility_monthly VALUES (" + ",".join("?" * 13) + ")", facility_rows)

    print(str(args.out))
    print(f"  obligor_monthly  {len(obligor_rows):>6} rows")
    print(f"  facility_monthly {len(facility_rows):>6} rows")

    # Grain must be unique, or every downstream factsheet is wrong.
    for table, keys in (("obligor_monthly", "obligor_id, observation_date"),
                        ("facility_monthly", "obligor_id, facility_id, observation_date")):
        dupes = con.execute(
            f"SELECT count(*) FROM (SELECT {keys} FROM {table} "
            f"GROUP BY {keys} HAVING count(*) > 1)").fetchone()[0]
        print(f"  {table} duplicate grain keys: {dupes}   (must be 0)")

    print()
    print("Corporate + SAMA obligors (use these in the curl examples):")
    for row in con.execute(
        "SELECT DISTINCT obligor_id FROM obligor_monthly "
        "WHERE portfolio='corporate' AND jurisdiction='SAMA' ORDER BY 1 LIMIT 5"
    ).fetchall():
        print(f"  {row[0]}")

    print()
    print("Point-in-time lag (data lands +5d, models run +10d after month end):")
    for as_of in ("2025-12-31", "2026-01-15"):
        visible = con.execute(
            "SELECT count(*) FROM obligor_monthly WHERE obligor_id='OBL-0008' "
            "AND data_cutoff_date <= ? AND model_run_date <= ?", [as_of, as_of]).fetchone()[0]
        print(f"  as at {as_of}: {visible} of 12 months visible for OBL-0008")
    con.close()


if __name__ == "__main__":
    main()
