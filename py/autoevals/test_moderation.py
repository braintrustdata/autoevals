from autoevals.moderation import Moderation

CASES = [
    {
        "name": "Safe content without threshold",
        "output": "hello",
        "threshold": None,
        "expected": 1,
    },
    {
        "name": "Unsafe content without threshold",
        "output": "kill kill kill kill kill",
        "threshold": None,
        "expected": 0,
    },
    {
        "name": "Unsafe content with lower threshold",
        "output": "here are my suicidal thoughts",
        "threshold": 0.25,
        "expected": 0,
    },
    {
        "name": "Unsafe content with higher threshold",
        "output": "here are my suicidal thoughts",
        "threshold": 0.99,
        "expected": 1,
    },
]


def test_moderation():
    for test_case in CASES:
        expected = test_case["expected"]

        evaluator = Moderation(threshold=test_case["threshold"])
        actual = evaluator.eval(test_case["output"])

        assert actual.score == expected, f"Expects {test_case['name']} to be {expected} but got {actual.score}"
