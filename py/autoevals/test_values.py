import pytest
from pytest import approx

from autoevals.list import ListContains
from autoevals.number import NumericDiff
from autoevals.string import LevenshteinScorer
from autoevals.value import ExactMatch


def test_levenshtein():
    cases = [
        ("", "", 1),
        ("", "a", 0),
        ("a", "", 0),
        ("a", "a", 1),
        ("a", "b", 0),
        ("ab", "ac", 0.5),
        ("ac", "bc", 0.5),
        ("abc", "axc", 0.66667),
        ("xabxcdxxefxgx", "1ab2cd34ef5g6", 0.53846),
    ]

    evaluator = LevenshteinScorer()
    for a, b, expected in cases:
        print(f"[{a}]", f"[{b}]", expected, evaluator(a, b))
        assert evaluator(a, b).score == approx(expected, abs=1e-4)


def test_numeric():
    cases = [(0, 0, 1), (0, 1, 0), (1, 2, 0.66667), (1.0, 2.0, 0.66667), (-1, 2, 0)]

    evaluator = NumericDiff()
    for a, b, expected in cases:
        print(f"[{a}]", f"[{b}]", expected, evaluator(a, b))
        assert evaluator(a, b).score == approx(expected, abs=1e-4)


def test_list_contains():
    cases = [
        [[], [], 1],
        [[0], [], 0],
        [[], [0], 0],
        [["a"], ["a"], 1],
        [["a"], ["a", "b"], 0.5],
        [["a", "b"], ["a"], 0.5],
        [
            [
                "workspaces",
                "section",
                "view",
                "others",
                "workspace",
                "team",
                "pinning",
            ],
            ["starred", "multiple different workspaces", "shortcuts"],
            0.1218,
        ],
        [
            ["starred", "multiple different workspaces", "shortcuts"],
            [
                "workspaces",
                "section",
                "view",
                "others",
                "workspace",
                "team",
                "pinning",
            ],
            0.1218,
        ],
    ]

    for output, expected, expected_score in cases:
        assert ListContains(pairwise_evaluator=LevenshteinScorer())(output, expected).score == approx(
            expected_score, abs=1e-4
        ), (output, expected, expected_score)

    assert (
        ListContains(pairwise_evaluator=LevenshteinScorer(), allow_extra_entities=True)(["a", "b"], ["a"]).score == 1
    )


def test_exact_match():
    cases = [
        ["hello", "hello", 1],
        ["hello", "world", 0],
        [123, 123, 1],
        [123, "123", 1],
        [{"a": 1, "b": 2}, {"a": 1, "b": 2}, 1],
        [{"a": 1, "b": 2}, {"a": 1, "b": 3}, 0],
        [[1, 2, 3], [1, 2, 3], 1],
        [[1, 2, 3], [3, 2, 1], 0],
        [{"a": 1, "b": 2}, {"b": 2, "a": 1}, 0],  # Order matters
        [{"a": 1, "b": 2}, '{"a": 1, "b": 2}', 1],  # String representation matches dict
        [{"a": 1, "b": 2}, '{"a":1, "b":2}', 1],  # String representation matches dict
        [{"a": 1, "b": 2}, '{"b": 2, "a": 1}', 0],
        [{"a": 1, "b": 2}, {"b": 2, "a": 1, "c": 3}, 0],  # Extra key, not equal
        [None, None, 1],
        [None, "None", 1],
    ]

    for output, expected, expected_score in cases:
        assert ExactMatch()(output, expected).score == approx(expected_score, abs=1e-4), (
            output,
            expected,
            expected_score,
        )
