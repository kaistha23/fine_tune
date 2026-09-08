"""Rating scale, issuer profile, and the prompt/answer text contract.

Every stage of the pipeline (data generation, training, evaluation, serving) formats
prompts through this module so the fine-tuned model never sees a layout it wasn't
trained on.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Seven major categories rather than the full 21-notch scale. With a few thousand
# synthetic examples the notched scale leaves too few samples per class to learn the
# tails; see README for how to extend once real data is available.
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

# BBB and better is investment grade; BB and below is speculative grade.
LOWEST_INVESTMENT_GRADE_INDEX = RATINGS.index("BBB")

SYSTEM_PROMPT = (
    "You are a corporate credit rating analyst. Given an issuer's financial and "
    "business profile, respond with a rating on the scale AAA, AA, A, BBB, BB, B, "
    "CCC followed by a brief rationale."
)

CYCLICALITY_LABELS = {1: "Very low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very high"}
POSITION_LABELS = {1: "Weak", 2: "Below average", 3: "Average", 4: "Strong", 5: "Dominant"}


@dataclass(frozen=True)
class Issuer:
    name: str
    sector: str
    revenue_musd: float
    revenue_growth: float
    ebitda_margin: float
    debt_to_ebitda: float
    interest_coverage: float
    current_ratio: float
    cyclicality: int
    competitive_position: int
    rate_environment: str

    def as_dict(self) -> dict:
        return asdict(self)


def format_profile(issuer: Issuer) -> str:
    """Render an issuer as the user-turn prompt."""
    return "\n".join(
        [
            "Assess the credit rating for the following issuer.",
            "",
            f"Company: {issuer.name}",
            f"Sector: {issuer.sector}",
            f"Revenue: ${issuer.revenue_musd:,.0f}M",
            f"Revenue growth: {issuer.revenue_growth * 100:.1f}%",
            f"EBITDA margin: {issuer.ebitda_margin * 100:.1f}%",
            f"Total debt / EBITDA: {issuer.debt_to_ebitda:.2f}x",
            f"EBIT / interest expense: {issuer.interest_coverage:.2f}x",
            f"Current ratio: {issuer.current_ratio:.2f}",
            f"Industry cyclicality: {CYCLICALITY_LABELS[issuer.cyclicality]} "
            f"({issuer.cyclicality}/5)",
            f"Competitive position: {POSITION_LABELS[issuer.competitive_position]} "
            f"({issuer.competitive_position}/5)",
            f"Rate environment: {issuer.rate_environment}",
        ]
    )


def format_answer(rating: str, rationale: str) -> str:
    return f"Rating: {rating}\n\nRationale: {rationale}"


# Longest alternatives first so "BBB" is never truncated to "BB".
_RATING_ALTERNATION = "|".join(sorted(RATINGS, key=len, reverse=True))
_LABELLED_RATING = re.compile(rf"rating\s*[:\-]?\s*({_RATING_ALTERNATION})\b", re.IGNORECASE)
_BARE_RATING = re.compile(rf"\b({_RATING_ALTERNATION})\b")


def parse_rating(text: str) -> str | None:
    """Extract the rating from a model response, or None if it emitted nothing usable.

    Prefers an explicit "Rating: X" label and falls back to the first bare rating token,
    so a model that drifts from the trained format still gets scored rather than silently
    counting as wrong.
    """
    match = _LABELLED_RATING.search(text)
    if match is None:
        match = _BARE_RATING.search(text)
    return match.group(1).upper() if match else None


def rating_index(rating: str) -> int:
    return RATINGS.index(rating)


def is_investment_grade(rating: str) -> bool:
    return rating_index(rating) <= LOWEST_INVESTMENT_GRADE_INDEX


def to_chat_example(issuer: Issuer, rating: str, rationale: str) -> dict:
    """Build one mlx-lm chat-format training record."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_profile(issuer)},
            {"role": "assistant", "content": format_answer(rating, rationale)},
        ]
    }
