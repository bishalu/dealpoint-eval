"""Pure-Python agreement statistics for the M5 judge calibration package.

No I/O, no numpy dependency in the maths (numpy is available elsewhere in
this repo but these functions stay plain and legible, matching
`dealpoint.eval.scorers`'s framework-free stance -- spec deliverable 5 / D3).

Every function guards its degenerate inputs (n < 2, all-None, zero
variance, a single distinct rating) by returning `None`, never `NaN` and
never raising. `None` means "not computable", and callers must render it as
JSON `null` with an accompanying `n`, never coerce it to 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations


def _paired_non_none(xs: Sequence, ys: Sequence) -> tuple[list, list]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if not pairs:
        return [], []
    a, b = zip(*pairs)
    return list(a), list(b)


def rank_average(xs: Sequence[float]) -> list[float]:
    """Average-rank transform (ties get the mean of their tied rank positions).

    1-indexed ranks, ascending. E.g. [1, 2, 2, 4] -> [1.0, 2.5, 2.5, 4.0].
    """
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(xs: Sequence[float | None], ys: Sequence[float | None]) -> float | None:
    """Pearson correlation, or `None` if n < 2 or either series has zero variance."""
    a, b = _paired_non_none(xs, ys)
    n = len(a)
    if n < 2:
        return None
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a == 0 or var_b == 0:
        return None
    return cov / (var_a**0.5 * var_b**0.5)


def spearman(xs: Sequence[float | None], ys: Sequence[float | None]) -> float | None:
    """Spearman's rho: Pearson over average ranks. `None` on the same degenerate cases."""
    a, b = _paired_non_none(xs, ys)
    if len(a) < 2:
        return None
    return pearson(rank_average(a), rank_average(b))


def exact_agreement(a: Sequence, b: Sequence) -> float | None:
    """Fraction of paired (non-None) ratings that are exactly equal. `None` if n == 0."""
    pa, pb = _paired_non_none(a, b)
    if not pa:
        return None
    return sum(1 for x, y in zip(pa, pb) if x == y) / len(pa)


def mean_abs_diff(a: Sequence, b: Sequence) -> float | None:
    """Mean absolute difference over paired (non-None) ratings. `None` if n == 0."""
    pa, pb = _paired_non_none(a, b)
    if not pa:
        return None
    return sum(abs(x - y) for x, y in zip(pa, pb)) / len(pa)


def quadratic_weighted_kappa(
    a: Sequence[int | None], b: Sequence[int | None], min_rating: int = 1, max_rating: int = 5
) -> float | None:
    """Quadratic-weighted Cohen's kappa over paired (non-None) integer ratings.

    Returns `None` when n < 2, or when both raters used only a single
    distinct rating each (the expected-agreement-by-chance matrix degenerates
    to zero variance, making kappa undefined rather than the conventional
    0/0 -> 1.0 or NaN).
    """
    pa, pb = _paired_non_none(a, b)
    n = len(pa)
    if n < 2:
        return None

    n_ratings = max_rating - min_rating + 1
    if n_ratings < 2:
        return None

    # Observed matrix
    observed = [[0] * n_ratings for _ in range(n_ratings)]
    hist_a = [0] * n_ratings
    hist_b = [0] * n_ratings
    for x, y in zip(pa, pb):
        ix = int(x) - min_rating
        iy = int(y) - min_rating
        if not (0 <= ix < n_ratings and 0 <= iy < n_ratings):
            continue
        observed[ix][iy] += 1
        hist_a[ix] += 1
        hist_b[iy] += 1

    if sum(1 for h in hist_a if h > 0) < 2 and sum(1 for h in hist_b if h > 0) < 2:
        # Both raters used a single rating each: kappa is undefined, not 1.0.
        return None

    weights = [
        [((i - j) ** 2) / ((n_ratings - 1) ** 2) for j in range(n_ratings)]
        for i in range(n_ratings)
    ]

    expected = [
        [hist_a[i] * hist_b[j] / n for j in range(n_ratings)] for i in range(n_ratings)
    ]

    numer = sum(
        weights[i][j] * observed[i][j] for i in range(n_ratings) for j in range(n_ratings)
    )
    denom = sum(
        weights[i][j] * expected[i][j] for i in range(n_ratings) for j in range(n_ratings)
    )
    if denom == 0:
        return None
    return 1.0 - (numer / denom)


def pairwise_judge_agreement(scores_by_judge: dict[str, dict[str, int | None]]) -> list[dict]:
    """One entry per unordered judge pair, over their commonly-scored packet ids.

    `scores_by_judge` maps judge_model -> {packet_id: score_or_None}. Each
    output entry carries `judge_a`, `judge_b`, `spearman`, `qwk`,
    `exact_agreement`, `mean_abs_diff`, `n` (the count of packets both judges
    scored non-None).
    """
    out: list[dict] = []
    judges = sorted(scores_by_judge)
    for judge_a, judge_b in combinations(judges, 2):
        a_scores = scores_by_judge[judge_a]
        b_scores = scores_by_judge[judge_b]
        common = sorted(set(a_scores) & set(b_scores))
        a_vals = [a_scores[pid] for pid in common]
        b_vals = [b_scores[pid] for pid in common]
        pa, _pb = _paired_non_none(a_vals, b_vals)
        out.append(
            {
                "judge_a": judge_a,
                "judge_b": judge_b,
                "spearman": spearman(a_vals, b_vals),
                "qwk": quadratic_weighted_kappa(a_vals, b_vals),
                "exact_agreement": exact_agreement(a_vals, b_vals),
                "mean_abs_diff": mean_abs_diff(a_vals, b_vals),
                "n": len(pa),
            }
        )
    return out


def correlate_with_grounded_accuracy(
    quality: Sequence[float | None], grounded: Sequence[bool | None]
) -> dict:
    """Pearson (point-biserial) and Spearman correlation of judged quality with
    `grounded_accuracy`, over indices where both are non-`None`.

    `grounded_accuracy` is `None` on CAP_HIT/EXECUTION_FAILED/abstained rows
    -- a large share of this project's data -- so pairing drops those
    indices rather than coercing them to 0/False. When n < 30 the result
    carries a `caveat` string.
    """
    pairs = [
        (q, float(g))
        for q, g in zip(quality, grounded)
        if q is not None and g is not None
    ]
    n = len(pairs)
    if n == 0:
        qs, gs = [], []
    else:
        qs, gs = zip(*pairs)
        qs, gs = list(qs), list(gs)
    result: dict = {
        "pearson": pearson(qs, gs) if n >= 2 else None,
        "spearman": spearman(qs, gs) if n >= 2 else None,
        "n": n,
    }
    if n < 30:
        result["caveat"] = f"n < 30 ({n}); correlation is under-powered."
    return result
