"""A skipped sub-score must not be averaged as if it were a zero.

`Score.score` is documented as "If the score is None, the evaluation is considered to be
skipped", and `JSONDiff.json_diff` filters those out before averaging. The dict branch then
divided by the filtered length, correctly ignoring the skip, while the list branch divided
by `max(len(o1), len(o2))`, which still counts it. The same skip was therefore ignored
inside an object and scored as a mismatch inside an array.

The dict branch also divided by `len(base_scores)` with no guard, so an object whose every
comparison was skipped raised ZeroDivisionError instead of reporting a skip.
"""

import pytest

from autoevals.json import JSONDiff
from autoevals.score import Score


class _SkipStrings:
    """Stands in for a string scorer that skips, e.g. an LLM judge that abstained."""

    def eval(self, output, expected=None, **kwargs):
        return Score(name="skipped", score=None)


class _ExactStrings:
    def eval(self, output, expected=None, **kwargs):
        return Score(name="exact", score=1 if output == expected else 0)


def _diff(**kwargs) -> JSONDiff:
    return JSONDiff(**kwargs)


class TestSkippedScoresAreNotCountedAsMismatches:
    def test_skipped_element_in_a_list_is_not_scored_as_zero(self):
        """One of two elements is skipped; the other matches, so the score is 1."""
        scorer = _diff(string_scorer=_SkipStringsForOne())
        assert scorer.json_diff(["skip", "same"], ["skip", "same"]) == 1

    def test_list_and_dict_treat_an_identical_skip_the_same_way(self):
        """The same pair of values, once in an array and once in an object."""
        scorer = _diff(string_scorer=_SkipStringsForOne())
        as_list = scorer.json_diff(["skip", "same"], ["skip", "same"])
        as_dict = scorer.json_diff({"a": "skip", "b": "same"}, {"a": "skip", "b": "same"})
        assert as_list == as_dict

    def test_a_fully_skipped_object_reports_a_skip_rather_than_raising(self):
        scorer = _diff(string_scorer=_SkipStrings())
        assert scorer.json_diff({"a": "x"}, {"a": "y"}) is None

    def test_a_fully_skipped_list_reports_a_skip_rather_than_raising(self):
        scorer = _diff(string_scorer=_SkipStrings())
        assert scorer.json_diff(["x"], ["y"]) is None

    def test_missing_elements_are_still_penalised(self):
        """Dropping skips from the denominator must not also drop real differences."""
        scorer = _diff(string_scorer=_ExactStrings())
        assert scorer.json_diff(["a"], ["a", "b"]) == 0.5

    def test_unskipped_lists_are_unchanged(self):
        scorer = _diff(string_scorer=_ExactStrings())
        assert scorer.json_diff(["a", "b"], ["a", "b"]) == 1
        assert scorer.json_diff(["a", "x"], ["a", "b"]) == 0.5

    def test_unskipped_dicts_are_unchanged(self):
        scorer = _diff(string_scorer=_ExactStrings())
        assert scorer.json_diff({"a": "1"}, {"a": "1"}) == 1
        assert scorer.json_diff({"a": "1", "b": "2"}, {"a": "1", "b": "3"}) == 0.5

    def test_empty_containers_still_score_one(self):
        scorer = _diff(string_scorer=_ExactStrings())
        assert scorer.json_diff({}, {}) == 1
        assert scorer.json_diff([], []) == 1


class _SkipStringsForOne:
    """Skips only the value "skip", so a single element of a pair is skipped."""

    def eval(self, output, expected=None, **kwargs):
        if output == "skip":
            return Score(name="skipped", score=None)
        return Score(name="exact", score=1 if output == expected else 0)
