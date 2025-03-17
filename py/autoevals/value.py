import json
from typing import Any

from braintrust_core.score import Score

from autoevals.partial import ScorerWithPartial


class ExactMatch(ScorerWithPartial):
    """
    A simple scorer that tests whether two values are equal. If the value is an object or array,
    it will be JSON-serialized and the strings compared for equality.
    """

    def _run_eval_sync(self, output, expected=None, **kwargs):
        maybe_object = needs_json(output) or needs_json(expected)
        output, expected = normalize_value(output, maybe_object), normalize_value(expected, maybe_object)
        score = 1 if output == expected else 0

        return Score(name=self._name(), score=score)


def needs_json(value: Any) -> bool:
    return isinstance(value, (dict, list))


def normalize_value(value: Any, maybe_object: bool) -> str:
    if needs_json(value):
        return json.dumps(value)

    try:
        if maybe_object:
            return json.dumps(json.loads(value))
    except json.JSONDecodeError:
        pass

    return str(value)
