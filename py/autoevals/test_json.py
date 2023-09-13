from pytest import approx

from autoevals.json import JSONDiff


def test_string_as_json():
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

    evaluator = JSONDiff()
    for a, b, expected in cases:
        print(f"[{a}]", f"[{b}]", expected, evaluator(a, b))
        assert evaluator(a, b).score == approx(expected, abs=1e-4)


def test_json():
    cases = [
        (None, None, 1),
        (None, "", 0),
        ([], {}, 0),
        ([], [], 1),
        ({}, {}, 1),
        ({"a": 1}, {"a": 1}, 1),
        ({"a": 1}, {"a": 2}, 0.66667),
        ({"a": 1}, ["a", 1], 0.5714285714285714),
        ({"a": 1}, {"b": {"a": 1}}, 0),
        ({"a": 1}, {"a": None}, 0),
        (
            {"mapping": {"a": "foo", "b": "bar"}},
            {"mapping": {"a": "Foo", "b": "Bar"}, "Extra": 5},
            0.33333333333333337,
        ),
    ]

    evaluator = JSONDiff()
    for a, b, expected in cases:
        print(f"[{a}]", f"[{b}]", expected, evaluator(a, b))
        assert evaluator(a, b).score == approx(expected, 1e-4)
