import math

import pytest

from autoevals import Score
from autoevals.consistency import (
    ConsistencyPlan,
    Rating,
    ScoreMapping,
    analyze_consistency,
    rating_from_score,
)


def plan(**kwargs):
    return ConsistencyPlan(score_name="judge", raters=("a", "b", "c"), categories=("no", "yes"), **kwargs)


def test_missing_manifest_cases_and_cells_are_retained():
    result = analyze_consistency([Rating("one", "a", "yes")], plan=plan(cases=("one", "two")))
    coverage = result["groups"][0]["coverage"]
    assert coverage["expected_cells"] == 6
    assert coverage["absent_cells"] == 5
    assert coverage["scored_cells"] == 1
    assert result["groups"][0]["cases"][0]["consensus"] is None


def test_duplicate_cells_are_invalid_even_when_identical():
    with pytest.raises(ValueError, match="duplicate"):
        analyze_consistency([Rating("one", "a", "yes")] * 2, plan=plan())


def test_score_adapter_selects_explicit_exact_values_and_omits_metadata():
    mapping = ScoreMapping("judge", ((0, "no"), (1, "yes")))
    rating = rating_from_score("one", "a", Score("judge", 1, {"secret": object()}), mapping=mapping)
    assert rating.category == "yes"
    assert "secret" not in repr(rating)
    assert rating_from_score("one", "a", Score("judge", None), mapping=mapping).status == "missing"
    with pytest.raises(ValueError, match="mapped"):
        rating_from_score("one", "a", Score("judge", 0.5), mapping=mapping)


@pytest.mark.parametrize("value", [math.nan, math.inf, True, "1", -1, 2])
def test_score_adapter_rejects_invalid_numeric_shapes(value):
    mapping = ScoreMapping("judge", ((0, "no"), (1, "yes")))
    with pytest.raises(ValueError):
        rating_from_score("one", "a", {"name": "judge", "score": value}, mapping=mapping)


def test_unknown_rater_or_category_rejects():
    with pytest.raises(ValueError, match="rater"):
        analyze_consistency([Rating("one", "other", "yes")], plan=plan())
    with pytest.raises(ValueError, match="category"):
        analyze_consistency([Rating("one", "a", "unknown")], plan=plan())
