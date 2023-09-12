from autoevals.number import NumericDifference
from autoevals.string import LevenshteinScorer


def test_levenshtein():
    cases = [
        ("", "", 1),
        ("", "a", 0),
        ("a", "", 0),
        ("a", "a", 1),
        ("a", "b", 0),
        ("ab", "ac", 0.5),
        ("ac", "bc", 0.5),
        ("abc", "axc", 0.6666666666666667),
        ("xabxcdxxefxgx", "1ab2cd34ef5g6", 0.5384615384615384),
    ]

    evaluator = LevenshteinScorer()
    for a, b, expected in cases:
        print(f"[{a}]", f"[{b}]", expected, evaluator(a, b))
        assert evaluator(a, b).score == expected


def test_numeric():
    cases = [(0, 0, 1), (0, 1, 0), (1, 2, 0.5), (1.0, 2.0, 0.5)]

    evaluator = NumericDifference()
    for a, b, expected in cases:
        print(f"[{a}]", f"[{b}]", expected, evaluator(a, b))
        assert evaluator(a, b).score == expected
