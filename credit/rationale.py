"""Compose the rationale text for an already-decided rating.

This is step B of the two-step data design. The rating is fixed before this module runs,
and the wording is assembled from the same factor scores the scorecard used, so a
rationale can never argue for a different rating than the label it accompanies.

The phrasing is templated rather than LLM-generated. That trades some linguistic variety
for a guarantee of label/rationale consistency and removes any API dependency from data
generation. See README for how to swap in an LLM pass if you want richer prose.
"""

from __future__ import annotations

import numpy as np

from .schema import Issuer, is_investment_grade

# Factors below this score read as credit negatives, above as credit positives.
_WEAK_THRESHOLD = 45.0
_STRONG_THRESHOLD = 62.0

# How much each factor moves the rating, mirroring the scorecard weights. Used to pick
# which drivers are worth mentioning.
_SALIENCE = {
    "leverage": 0.24, "coverage": 0.20, "margin": 0.13, "scale": 0.10,
    "cyclicality": 0.10, "position": 0.10, "liquidity": 0.07, "growth": 0.06,
}


def _describe(factor: str, issuer: Issuer, weak: bool) -> str:
    """Render one driver as a noun phrase citing the underlying figure."""
    v = issuer
    if factor == "leverage":
        return (
            f"elevated leverage of {v.debt_to_ebitda:.1f}x debt/EBITDA" if weak
            else f"conservative leverage of {v.debt_to_ebitda:.1f}x debt/EBITDA"
        )
    if factor == "coverage":
        return (
            f"thin interest coverage of {v.interest_coverage:.1f}x" if weak
            else f"comfortable interest coverage of {v.interest_coverage:.1f}x"
        )
    if factor == "margin":
        return (
            f"a compressed EBITDA margin of {v.ebitda_margin * 100:.0f}%" if weak
            else f"a healthy EBITDA margin of {v.ebitda_margin * 100:.0f}%"
        )
    if factor == "liquidity":
        return (
            f"tight liquidity at a {v.current_ratio:.1f} current ratio" if weak
            else f"solid liquidity at a {v.current_ratio:.1f} current ratio"
        )
    if factor == "scale":
        return (
            f"limited scale at ${v.revenue_musd:,.0f}M of revenue" if weak
            else f"substantial scale at ${v.revenue_musd:,.0f}M of revenue"
        )
    if factor == "growth":
        return (
            f"weak top-line momentum of {v.revenue_growth * 100:.1f}%" if weak
            else f"solid top-line growth of {v.revenue_growth * 100:.1f}%"
        )
    if factor == "cyclicality":
        return (
            f"high cyclicality in the {v.sector.lower()} sector" if weak
            else f"the stability of {v.sector.lower()} sector cash flows"
        )
    if factor == "position":
        return (
            "a weak competitive position" if weak
            else "a strong competitive position"
        )
    raise ValueError(f"unknown factor: {factor}")


def _join(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    return f"{', '.join(phrases[:-1])} and {phrases[-1]}"


def _sentence(frame: str, drivers: str) -> str:
    """Fill a frame and capitalize it — driver phrases are lowercase noun phrases."""
    text = frame.format(drivers=drivers)
    return text[0].upper() + text[1:]


_NEGATIVE_FRAMES = [
    "{drivers} constrain the credit profile.",
    "The rating is held back by {drivers}.",
    "{drivers} are the primary constraints on the rating.",
    "Credit quality is limited by {drivers}.",
]
_POSITIVE_FRAMES = [
    "{drivers} support the rating.",
    "The profile benefits from {drivers}.",
    "{drivers} underpin credit quality.",
    "Supporting the rating are {drivers}.",
]
_OFFSET_FRAMES = [
    "{drivers} provide partial offset.",
    "This is partly offset by {drivers}.",
    "Offsetting factors include {drivers}.",
    "{drivers} temper these concerns.",
]
_RESIDUAL_FRAMES = [
    "Against this, {drivers} remain a constraint.",
    "{drivers} weigh against the profile.",
    "Balancing that, {drivers} limit further upside.",
]
_RATE_NOTES = {
    "Rising": [
        "A rising rate environment adds refinancing pressure given the debt load.",
        "Rising rates raise the cost of servicing this capital structure.",
    ],
    "Falling": [
        "A falling rate environment eases refinancing risk.",
        "Declining rates modestly improve the funding outlook.",
    ],
    "Stable": [],
}


def compose_rationale(
    issuer: Issuer,
    rating: str,
    factors: dict[str, float],
    rng: np.random.Generator,
) -> str:
    """Build a 2-4 sentence rationale from the factor scores behind `rating`."""
    ranked = sorted(
        factors.items(),
        key=lambda kv: _SALIENCE[kv[0]] * abs(kv[1] - 50.0),
        reverse=True,
    )
    weaknesses = [f for f, score in ranked if score < _WEAK_THRESHOLD][:2]
    strengths = [f for f, score in ranked if score > _STRONG_THRESHOLD][:2]

    sentences: list[str] = []
    if weaknesses:
        drivers = _join([_describe(f, issuer, weak=True) for f in weaknesses])
        sentences.append(_sentence(str(rng.choice(_NEGATIVE_FRAMES)), drivers))
        if strengths:
            drivers = _join([_describe(f, issuer, weak=False) for f in strengths])
            sentences.append(_sentence(str(rng.choice(_OFFSET_FRAMES)), drivers))
    elif strengths:
        drivers = _join([_describe(f, issuer, weak=False) for f in strengths])
        sentences.append(_sentence(str(rng.choice(_POSITIVE_FRAMES)), drivers))
        residual = [f for f, score in ranked if score < 50.0][:1]
        if residual:
            drivers = _join([_describe(f, issuer, weak=True) for f in residual])
            sentences.append(_sentence(str(rng.choice(_RESIDUAL_FRAMES)), drivers))
    else:
        sentences.append(
            f"The profile is broadly balanced, with no factor materially away from the "
            f"{issuer.sector.lower()} sector norm."
        )

    rate_notes = _RATE_NOTES[issuer.rate_environment]
    if rate_notes and issuer.debt_to_ebitda > 3.0:
        sentences.append(str(rng.choice(rate_notes)))

    grade = "investment grade" if is_investment_grade(rating) else "speculative grade"
    sentences.append(f"On balance this supports a {rating} rating, in {grade} territory.")
    return " ".join(sentences)
