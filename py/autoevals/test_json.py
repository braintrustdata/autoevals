from pytest import approx

from autoevals.json import JSONDiff, ValidJSON
from autoevals.number import NumericDiff
from autoevals.value import ExactMatch


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


def test_valid_json():
    cases = [
        ("1", 0, None),
        ('{ "a": 1, "b": "hello" }', 1, None),
        ('[{ "a": 1 }]', 1, None),
        ('[{ "a": 1 }', 0, None),
        ('{ "mapping": { "a": "foo", "b": "bar" }, "extra": 4 }', 1, None),
        ('{ mapping: { "a": "foo", "b": "bar" }, "extra": 4 }', 0, None),
        (
            '{ "a": "1" }',
            1,
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
        ),
        (
            '{"a": "1", "b": "1"}',
            0,
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "number"},
                },
                "required": ["a"],
            },
        ),
        (
            '[{"a": "1"}, {"a": "1", "b": 22}]',
            1,
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"type": "number"},
                    },
                    "required": ["a"],
                },
                "uniqueItems": True,
            },
        ),
        (
            {"a": "1", "b": "1"},
            1,
            None,
        ),
        (
            [{"a": "1"}, {"a": "1", "b": 22}],
            1,
            None,
        ),
        (
            100,
            0,
            None,
        ),
        (
            # This is technically ambiguous, because it _could_ be the valid parsed JSON value
            # or an unparsed, invalid JSON value. However, since structured outputs _only_ return
            # JSON values, we can safely assume that any strings are unparsed values.
            "100",
            0,
            None,
        ),
    ]

    evaluator = ValidJSON()
    for output, expected, schema in cases:
        print(f"[{output}]", expected)
        assert evaluator(output, schema).score == expected


def test_semantic_json():
    cases = [
        ('{"x": 1, "y": 2}', '{"y": 2, "x": 1}', 1),
        (
            '{"zs": ["a", "b"], "x": 1, "y": 2}',
            '{"y": 2, "zs": ["a", "b"], "x": 1}',
            1,
        ),
        (
            '{"o1": {"x": 1, "y": 2}}',
            '{"o1": {"y": 2, "x": 1}}',
            1,
        ),
        (
            '{"xs": [{"o1": {"x": 1, "y": [2]}}]}',
            '{"xs": [{"o1": {"y": [2], "x": 1}}]}',
            1,
        ),
        (
            '{"o1": {"x": 2, "y": 2}}',
            '{"o1": {"y": 2, "x": 1}}',
            0.83333,
        ),
        (
            {"o1": {"x": 2, "y": 2}},
            '{"o1": {"y": 2, "x": 1}}',
            0.83333,
        ),
        ('{"x": 1, "y": 2}', '{"x": 1, "z": 2}', 0.3333),
        ("[1, 2]", "[1, 2]", 1),
        ("[1, 2]", "[2, 1]", 0.66667),
    ]

    evaluator = JSONDiff()
    for a, b, expected in cases:
        for exact_number in [True, False]:
            score = evaluator(a, b, number_scorer=ExactMatch() if exact_number else NumericDiff()).score
            if not exact_number:
                assert abs(score - expected) < 0.0001
            else:
                assert round(score * 100) <= round(expected * 100)
