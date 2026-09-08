"""Deterministic rating scorecard and industry-conditioned issuer sampling.

This is step A of the two-step data design: the label for every synthetic issuer comes
from this scorecard, never from an LLM. That guarantees the training set encodes one
consistent mapping from financials to rating, which is the thing the fine-tune is
supposed to learn.

The factor weights and cut points are a plausible generic-corporate rubric in the style
of published agency criteria. They are not any agency's proprietary model.
"""

from __future__ import annotations

import math

import numpy as np

from .schema import RATINGS, Issuer

# (value, score) anchors, interpolated linearly and clamped at the ends. Scores are 0-100
# where 100 is the strongest credit outcome for that factor.
_ANCHORS = {
    # Scored on leverage already divided by the sector's tolerance multiplier.
    "leverage": [(0.5, 100), (1.5, 90), (2.5, 75), (3.5, 60), (4.5, 45), (6.0, 25), (8.0, 5), (12.0, 0)],
    "coverage": [(0.5, 0), (1.5, 20), (3.0, 45), (6.0, 70), (10.0, 85), (20.0, 100)],
    "margin": [(0.02, 5), (0.06, 25), (0.12, 50), (0.20, 70), (0.30, 88), (0.45, 100)],
    "liquidity": [(0.5, 5), (0.9, 30), (1.2, 55), (1.8, 80), (2.5, 95), (4.0, 100)],
    # x-axis is log10 of revenue in $M, i.e. ~$50M through ~$100B.
    "scale": [(1.7, 5), (2.5, 25), (3.0, 50), (3.7, 75), (4.3, 92), (5.0, 100)],
    "growth": [(-0.15, 5), (-0.05, 25), (0.0, 45), (0.05, 65), (0.12, 85), (0.25, 95)],
    "cyclicality": [(1, 100), (2, 80), (3, 60), (4, 35), (5, 10)],
    "position": [(1, 10), (2, 35), (3, 60), (4, 82), (5, 100)],
}

_WEIGHTS = {
    "leverage": 0.24,
    "coverage": 0.20,
    "margin": 0.13,
    "scale": 0.10,
    "cyclicality": 0.10,
    "position": 0.10,
    "liquidity": 0.07,
    "growth": 0.06,
}

# Composite score floor for each rating, best to worst.
_RATING_FLOORS = [("AAA", 88), ("AA", 78), ("A", 68), ("BBB", 56), ("BB", 42), ("B", 27)]

# Per sector: fixed cyclicality, a leverage tolerance multiplier (utilities and REITs
# sustain more debt at a given rating), and (mean, sd) for each sampled ratio.
INDUSTRIES = {
    "Utilities": dict(
        cyclicality=1, leverage_tolerance=1.60,
        debt_to_ebitda=(4.2, 1.1), interest_coverage=(3.5, 1.4), ebitda_margin=(0.30, 0.07),
        current_ratio=(1.00, 0.25), log_revenue=(3.7, 0.5), revenue_growth=(0.02, 0.03),
    ),
    "Technology": dict(
        cyclicality=3, leverage_tolerance=0.85,
        debt_to_ebitda=(1.6, 1.2), interest_coverage=(12.0, 7.0), ebitda_margin=(0.26, 0.10),
        current_ratio=(2.20, 0.70), log_revenue=(3.3, 0.8), revenue_growth=(0.14, 0.12),
    ),
    "Retail": dict(
        cyclicality=4, leverage_tolerance=0.95,
        debt_to_ebitda=(3.0, 1.3), interest_coverage=(5.0, 3.0), ebitda_margin=(0.09, 0.04),
        current_ratio=(1.30, 0.35), log_revenue=(3.5, 0.7), revenue_growth=(0.03, 0.06),
    ),
    "Industrials": dict(
        cyclicality=4, leverage_tolerance=1.00,
        debt_to_ebitda=(3.2, 1.2), interest_coverage=(5.5, 3.0), ebitda_margin=(0.14, 0.05),
        current_ratio=(1.50, 0.40), log_revenue=(3.4, 0.7), revenue_growth=(0.04, 0.06),
    ),
    "Healthcare": dict(
        cyclicality=2, leverage_tolerance=1.10,
        debt_to_ebitda=(3.0, 1.3), interest_coverage=(6.0, 3.5), ebitda_margin=(0.22, 0.08),
        current_ratio=(1.60, 0.50), log_revenue=(3.5, 0.7), revenue_growth=(0.07, 0.06),
    ),
    "Energy": dict(
        cyclicality=5, leverage_tolerance=1.05,
        debt_to_ebitda=(3.4, 1.6), interest_coverage=(5.0, 4.0), ebitda_margin=(0.25, 0.11),
        current_ratio=(1.20, 0.40), log_revenue=(3.8, 0.8), revenue_growth=(0.05, 0.18),
    ),
    "Consumer Staples": dict(
        cyclicality=1, leverage_tolerance=1.15,
        debt_to_ebitda=(2.9, 1.1), interest_coverage=(7.0, 3.5), ebitda_margin=(0.17, 0.06),
        current_ratio=(1.30, 0.35), log_revenue=(3.7, 0.7), revenue_growth=(0.03, 0.04),
    ),
    "Telecom": dict(
        cyclicality=2, leverage_tolerance=1.50,
        debt_to_ebitda=(3.8, 1.2), interest_coverage=(4.0, 2.0), ebitda_margin=(0.33, 0.08),
        current_ratio=(0.90, 0.30), log_revenue=(3.9, 0.6), revenue_growth=(0.01, 0.04),
    ),
    "Real Estate": dict(
        cyclicality=3, leverage_tolerance=1.80,
        debt_to_ebitda=(6.0, 1.8), interest_coverage=(3.0, 1.5), ebitda_margin=(0.40, 0.12),
        current_ratio=(1.10, 0.40), log_revenue=(3.2, 0.7), revenue_growth=(0.04, 0.07),
    ),
    "Materials": dict(
        cyclicality=5, leverage_tolerance=1.00,
        debt_to_ebitda=(3.1, 1.3), interest_coverage=(5.5, 3.5), ebitda_margin=(0.16, 0.07),
        current_ratio=(1.70, 0.50), log_revenue=(3.5, 0.7), revenue_growth=(0.03, 0.10),
    ),
}

RATE_ENVIRONMENTS = ["Falling", "Stable", "Rising"]

_NAME_PREFIXES = [
    "Northwind", "Redstone", "Halcyon", "Blue Harbor", "Ironbridge", "Cascade",
    "Meridian", "Fairmont", "Cobalt", "Silverline", "Granite Peak", "Aldermere",
    "Westbrook", "Larkspur", "Vantage", "Kestrel", "Beacon", "Thornhill",
    "Copperfield", "Sable", "Winterly", "Ashford", "Pinnacle", "Drayton",
]
_NAME_SUFFIXES = {
    "Utilities": ["Power", "Energy Systems", "Utilities", "Grid"],
    "Technology": ["Systems", "Software", "Labs", "Technologies"],
    "Retail": ["Retail Group", "Stores", "Markets", "Brands"],
    "Industrials": ["Manufacturing", "Industries", "Engineering", "Works"],
    "Healthcare": ["Health", "Medical", "Biosciences", "Care Group"],
    "Energy": ["Petroleum", "Resources", "Exploration", "Midstream"],
    "Consumer Staples": ["Foods", "Consumer Products", "Beverages", "Household"],
    "Telecom": ["Communications", "Telecom", "Networks", "Broadband"],
    "Real Estate": ["Properties", "Realty Trust", "Estates", "Property Group"],
    "Materials": ["Chemicals", "Materials", "Mining", "Metals"],
}
_NAME_FORMS = ["{p} {s} Corp", "{p} {s}", "{p} {s} Inc", "{p} {s} Holdings", "{p} {s} Group"]


def _interpolate(value: float, anchors: list[tuple[float, float]]) -> float:
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    return float(np.interp(value, xs, ys))


def score_issuer(issuer: Issuer) -> tuple[float, dict[str, float]]:
    """Return the 0-100 composite score and the per-factor scores behind it."""
    tolerance = INDUSTRIES[issuer.sector]["leverage_tolerance"]
    factors = {
        "leverage": _interpolate(issuer.debt_to_ebitda / tolerance, _ANCHORS["leverage"]),
        "coverage": _interpolate(issuer.interest_coverage, _ANCHORS["coverage"]),
        "margin": _interpolate(issuer.ebitda_margin, _ANCHORS["margin"]),
        "liquidity": _interpolate(issuer.current_ratio, _ANCHORS["liquidity"]),
        "scale": _interpolate(math.log10(max(issuer.revenue_musd, 1.0)), _ANCHORS["scale"]),
        "growth": _interpolate(issuer.revenue_growth, _ANCHORS["growth"]),
        "cyclicality": _interpolate(issuer.cyclicality, _ANCHORS["cyclicality"]),
        "position": _interpolate(issuer.competitive_position, _ANCHORS["position"]),
    }
    composite = sum(factors[name] * weight for name, weight in _WEIGHTS.items())
    composite += _rate_adjustment(issuer)
    return float(np.clip(composite, 0.0, 100.0)), factors


def _rate_adjustment(issuer: Issuer) -> float:
    """Rising rates hurt, and hurt more the more leveraged the issuer is."""
    if issuer.rate_environment == "Rising":
        return -(1.0 + 0.6 * max(0.0, issuer.debt_to_ebitda - 3.0))
    if issuer.rate_environment == "Falling":
        return 0.8
    return 0.0


def composite_to_rating(composite: float) -> str:
    for rating, floor in _RATING_FLOORS:
        if composite >= floor:
            return rating
    return RATINGS[-1]


def rate_issuer(issuer: Issuer, rng: np.random.Generator | None = None) -> tuple[str, dict[str, float]]:
    """Assign a rating, optionally with a touch of analyst-judgment noise.

    Without the noise term the mapping is perfectly separable, which is unrealistic and
    lets the model latch onto exact cut points instead of learning the general ordering.
    """
    composite, factors = score_issuer(issuer)
    if rng is not None:
        composite = float(np.clip(composite + rng.normal(0.0, 2.5), 0.0, 100.0))
    return composite_to_rating(composite), factors


def _clipped_normal(rng, mean_sd: tuple[float, float], low: float, high: float, shift: float = 0.0) -> float:
    mean, sd = mean_sd
    return float(np.clip(rng.normal(mean + shift, sd), low, high))


_POSITION_BASE_P = np.array([0.08, 0.20, 0.37, 0.27, 0.08])
_POSITION_TILT = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def sample_issuer(
    rng: np.random.Generator, sector: str | None = None, stress: float = 0.0
) -> Issuer:
    """Draw an issuer from its sector's distribution, shifted by `stress` in [-1, 1].

    Sector distributions alone cluster almost everything in A through BB, leaving the
    AAA and CCC tails effectively unpopulated. `stress` slides the whole profile toward
    distress (positive) or strength (negative) so the generated set spans the full scale.
    """
    sector = sector or str(rng.choice(list(INDUSTRIES)))
    params = INDUSTRIES[sector]
    prefix = str(rng.choice(_NAME_PREFIXES))
    suffix = str(rng.choice(_NAME_SUFFIXES[sector]))
    name = str(rng.choice(_NAME_FORMS)).format(p=prefix, s=suffix)

    coverage_mean, coverage_sd = params["interest_coverage"]
    coverage_mean = max(0.3, coverage_mean * (1.0 - 0.55 * stress))

    position_p = _POSITION_BASE_P * np.exp(-stress * _POSITION_TILT * 0.7)
    position_p /= position_p.sum()

    return Issuer(
        name=name,
        sector=sector,
        revenue_musd=10 ** _clipped_normal(rng, params["log_revenue"], 1.6, 5.2, -0.50 * stress),
        revenue_growth=_clipped_normal(rng, params["revenue_growth"], -0.30, 0.60, -0.10 * stress),
        ebitda_margin=_clipped_normal(rng, params["ebitda_margin"], 0.01, 0.60, -0.06 * stress),
        debt_to_ebitda=_clipped_normal(rng, params["debt_to_ebitda"], 0.0, 14.0, 2.5 * stress),
        interest_coverage=_clipped_normal(rng, (coverage_mean, coverage_sd), 0.2, 40.0),
        current_ratio=_clipped_normal(rng, params["current_ratio"], 0.3, 5.0, -0.40 * stress),
        cyclicality=params["cyclicality"],
        competitive_position=int(rng.choice([1, 2, 3, 4, 5], p=position_p)),
        rate_environment=str(rng.choice(RATE_ENVIRONMENTS, p=[0.25, 0.45, 0.30])),
    )
