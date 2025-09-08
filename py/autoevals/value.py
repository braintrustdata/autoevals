"""Value comparison utilities for exact matching and normalization.

This module provides tools for exact value comparison with smart handling of different data types:

- ExactMatch: A scorer for exact value comparison
  - Handles primitive types (strings, numbers, etc.)
  - Smart `JSON` serialization for objects and arrays
  - Normalizes `JSON` strings for consistent comparison

Example:
```python
from autoevals import ExactMatch

# Simple value comparison
scorer = ExactMatch()
result = scorer.eval(
    output="hello",
    expected="hello"
)
print(result.score)  # 1.0 for exact match

# Object comparison (automatically normalized)
result = scorer.eval(
    output={"name": "John", "age": 30},
    expected='{"age": 30, "name": "John"}'  # Different order but same content
)
print(result.score)  # 1.0 for equivalent JSON

# Array comparison
result = scorer.eval(
    output=[1, 2, 3],
    expected="[1, 2, 3]"  # String or native types work
)
print(result.score)  # 1.0 for equivalent arrays
```
"""

import json
from typing import Any

from autoevals.partial import ScorerWithPartial

from .score import Score


class ExactMatch(ScorerWithPartial):
    """A scorer that tests for exact equality between values.

    This scorer handles various input types:
    - Primitive values (strings, numbers, etc.)
    - JSON objects (dicts) and arrays (lists)
    - JSON strings that can be parsed into objects/arrays

    The comparison process:
    1. Detects if either value is/might be a JSON object/array
    2. Normalizes both values (serialization if needed)
    3. Performs exact string comparison

    Args:
        output: Value to evaluate
        expected: Reference value to compare against

    Returns:
        Score object with:
        - score: 1.0 for exact match, 0.0 otherwise
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
