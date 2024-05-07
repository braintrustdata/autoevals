from pytest import approx

from autoevals.json import JSONDiff, ValidJSON


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
    ]

    evaluator = ValidJSON()
    for output, expected, schema in cases:
        print(f"[{output}]", expected)
        assert evaluator(output, schema).score == expected
