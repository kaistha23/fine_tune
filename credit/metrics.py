"""Ordinal rating metrics, shared by the CLI evaluator and the training UI.

Ratings are ordinal, so distance on the scale matters more than exact match: predicting
BBB when the truth is BB is a far smaller error than predicting AAA.
"""

from __future__ import annotations

from collections import Counter

from .schema import RATINGS, is_investment_grade, rating_index

Pair = tuple[str, str | None]


def compute_metrics(pairs: list[Pair]) -> dict:
    """pairs: (true_rating, predicted_rating_or_None)."""
    total = len(pairs)
    unparseable = sum(1 for _, pred in pairs if pred is None)
    scored = [(t, p) for t, p in pairs if p is not None]

    # Unparseable responses are failures, not free passes: `total` stays the denominator.
    exact = sum(1 for t, p in scored if t == p)
    distances = [abs(rating_index(t) - rating_index(p)) for t, p in scored]
    within_one = sum(1 for d in distances if d <= 1)
    ig_correct = sum(1 for t, p in scored if is_investment_grade(t) == is_investment_grade(p))

    per_class = {}
    f1s = []
    for rating in RATINGS:
        tp = sum(1 for t, p in scored if t == rating and p == rating)
        fp = sum(1 for t, p in scored if t != rating and p == rating)
        fn = sum(1 for t, p in scored if t == rating and p != rating)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(1 for t, _ in pairs if t == rating)
        per_class[rating] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support
        }
        if support:
            f1s.append(f1)

    return {
        "n": total,
        "unparseable": unparseable,
        "exact_accuracy": exact / total if total else 0.0,
        "mean_notch_error": sum(distances) / len(distances) if distances else float("nan"),
        "median_notch_error": (
            sorted(distances)[len(distances) // 2] if distances else float("nan")
        ),
        "within_one_notch": within_one / total if total else 0.0,
        "ig_hy_accuracy": ig_correct / total if total else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
    }


def confusion_rows(pairs: list[Pair]) -> list[list]:
    """Confusion matrix as rows for a table: true rating, then one column per prediction."""
    counts = Counter(pairs)
    rows = []
    for true_rating in RATINGS:
        row: list = [true_rating]
        row.extend(counts[(true_rating, p)] for p in RATINGS)
        row.append(counts[(true_rating, None)])
        rows.append(row)
    return rows


CONFUSION_HEADERS = ["true \\ pred", *RATINGS, "?"]


def format_report(metrics: dict, pairs: list[Pair]) -> str:
    lines = [
        "=" * 62,
        "RESULTS",
        "=" * 62,
        f"  Examples scored         {metrics['n']}",
        f"  Unparseable responses   {metrics['unparseable']}",
        f"  Exact bucket accuracy   {metrics['exact_accuracy']:.1%}",
        f"  Within-1-notch accuracy {metrics['within_one_notch']:.1%}",
        f"  Mean notch error        {metrics['mean_notch_error']:.3f}   <- primary metric",
        f"  Median notch error      {metrics['median_notch_error']}",
        f"  IG/HY crossover acc.    {metrics['ig_hy_accuracy']:.1%}",
        f"  Macro F1                {metrics['macro_f1']:.3f}",
        "",
        "Per-class:",
        f"  {'rating':<8}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}",
    ]
    for rating in RATINGS:
        c = metrics["per_class"][rating]
        lines.append(
            f"  {rating:<8}{c['precision']:>8.2f}{c['recall']:>8.2f}"
            f"{c['f1']:>8.2f}{c['support']:>9}"
        )

    lines += ["", "Confusion matrix (rows = true, cols = predicted, '?' = unparseable):"]
    lines.append("  " + "".join(f"{h:>6}" for h in CONFUSION_HEADERS[1:]).rjust(8))
    for row in confusion_rows(pairs):
        lines.append(f"  {row[0]:<6}" + "".join(f"{v:>6}" for v in row[1:]))
    return "\n".join(lines)
