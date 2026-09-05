"""Agreement statistics unit-tested on synthetic scores (spec DoD check 4)."""

from __future__ import annotations

import pytest

from dealpoint.eval.agreement import (
    correlate_with_grounded_accuracy,
    exact_agreement,
    mean_abs_diff,
    pairwise_judge_agreement,
    pearson,
    quadratic_weighted_kappa,
    rank_average,
    spearman,
)

pytestmark = pytest.mark.gate_m5


def test_rank_average_no_ties():
    assert rank_average([10, 20, 30]) == [1.0, 2.0, 3.0]


def test_rank_average_with_ties():
    # [1, 2, 2, 4] -> ranks [1, 2.5, 2.5, 4]
    assert rank_average([1, 2, 2, 4]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_perfect_monotone_agreement_is_one():
    xs = [1, 2, 3, 4, 5]
    ys = [10, 20, 30, 40, 50]
    assert spearman(xs, ys) == pytest.approx(1.0)


def test_spearman_perfect_reversal_is_negative_one():
    xs = [1, 2, 3, 4, 5]
    ys = [5, 4, 3, 2, 1]
    assert spearman(xs, ys) == pytest.approx(-1.0)


def test_pearson_perfect_linear_is_one():
    xs = [1, 2, 3, 4]
    ys = [2, 4, 6, 8]
    assert pearson(xs, ys) == pytest.approx(1.0)


def test_quadratic_weighted_kappa_identical_raters_is_one():
    a = [1, 2, 3, 4, 5, 3, 2, 1]
    b = list(a)
    assert quadratic_weighted_kappa(a, b) == pytest.approx(1.0)


def test_quadratic_weighted_kappa_hand_worked_example():
    # Hand-worked confusion matrix, ratings 1..3 (min=1, max=3), n=6:
    #   a = [1, 1, 2, 2, 3, 3]
    #   b = [1, 2, 2, 3, 3, 1]
    # Observed matrix O (rows=a, cols=b), indices 0..2 for ratings 1..3:
    #   O = [[1,1,0],
    #        [0,1,1],
    #        [1,0,1]]
    # hist_a = [2,2,2], hist_b = [2,2,2], n=6
    # weights w[i][j] = (i-j)^2 / (n_ratings-1)^2 = (i-j)^2/4
    #   w = [[0, 0.25, 1], [0.25, 0, 0.25], [1, 0.25, 0]]
    # E[i][j] = hist_a[i]*hist_b[j]/n = 2*2/6 = 2/3 for all i,j
    # numerator = sum(w*O) = 0*1 + 0.25*1 + 1*0 + 0.25*0 + 0*1 + 0.25*1 + 1*1 + 0.25*0 + 0*1
    #           = 0.25 + 0.25 + 1 = 1.5
    # denominator = sum(w*E) = (2/3) * sum(w) = (2/3) * (0+0.25+1+0.25+0+0.25+1+0.25+0)
    #             = (2/3) * 3.0 = 2.0
    # kappa = 1 - numerator/denominator = 1 - 1.5/2.0 = 1 - 0.75 = 0.25
    a = [1, 1, 2, 2, 3, 3]
    b = [1, 2, 2, 3, 3, 1]
    result = quadratic_weighted_kappa(a, b, min_rating=1, max_rating=3)
    assert result == pytest.approx(0.25, abs=1e-9)


def test_rank_average_ties_assert_vector_directly():
    assert rank_average([5, 5, 5]) == [2.0, 2.0, 2.0]
    assert rank_average([1, 1, 3, 3]) == [1.5, 1.5, 3.5, 3.5]


# --- degenerate guards: None, never NaN, never an exception -----------------


def test_pearson_degenerate_returns_none():
    assert pearson([], []) is None
    assert pearson([1], [1]) is None
    assert pearson([1, 1, 1], [1, 2, 3]) is None  # zero variance in xs


def test_spearman_degenerate_returns_none():
    assert spearman([], []) is None
    assert spearman([1], [2]) is None


def test_quadratic_weighted_kappa_degenerate_single_rating_both_raters():
    # both raters used a single, identical rating throughout -> undefined, not 1.0
    assert quadratic_weighted_kappa([3, 3, 3], [3, 3, 3]) is None


def test_quadratic_weighted_kappa_too_few_returns_none():
    assert quadratic_weighted_kappa([1], [2]) is None
    assert quadratic_weighted_kappa([], []) is None


def test_exact_agreement_and_mean_abs_diff_degenerate():
    assert exact_agreement([], []) is None
    assert mean_abs_diff([], []) is None


def test_exact_agreement_and_mean_abs_diff_basic():
    a = [1, 2, 3, 4]
    b = [1, 2, 4, 4]
    assert exact_agreement(a, b) == pytest.approx(0.75)
    assert mean_abs_diff(a, b) == pytest.approx(0.25)


def test_pairwise_judge_agreement_handles_missing_scores():
    scores_by_judge = {
        "judge_a": {"p1": 4, "p2": 3, "p3": 5},
        "judge_b": {"p1": 4, "p2": None, "p3": 5},  # judge_b failed on p2
        "judge_c": {"p1": 3, "p2": 3},  # judge_c never scored p3
    }
    result = pairwise_judge_agreement(scores_by_judge)
    # 3 judges -> 3 pairs
    assert len(result) == 3
    by_pair = {(r["judge_a"], r["judge_b"]): r for r in result}
    ab = by_pair[("judge_a", "judge_b")]
    # p2 excluded because judge_b's score there is None -> only p1, p3 common+non-none
    assert ab["n"] == 2
    ac = by_pair[("judge_a", "judge_c")]
    # only p1, p2 are common keys at all (p3 absent from judge_c)
    assert ac["n"] == 2


def test_correlate_with_grounded_accuracy_pairs_only_non_none():
    quality = [4.0, 3.0, None, 5.0, 2.0]
    grounded = [True, False, True, None, True]
    # valid pairs: index0 (4.0,True), index1 (3.0,False), index4 (2.0,True) -> n=3
    result = correlate_with_grounded_accuracy(quality, grounded)
    assert result["n"] == 3
    assert "caveat" in result  # n < 30


def test_correlate_with_grounded_accuracy_empty_returns_none_not_nan():
    result = correlate_with_grounded_accuracy([], [])
    assert result["n"] == 0
    assert result["pearson"] is None
    assert result["spearman"] is None
