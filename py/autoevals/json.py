"""JSON evaluation scorers for comparing and validating JSON data.

This module provides scorers for working with JSON data:

- JSONDiff: Compare JSON objects for structural and content similarity
  - Handles nested structures, strings, numbers
  - Customizable with different scorers for string and number comparisons
  - Can automatically parse JSON strings

- ValidJSON: Validate if a string is valid JSON and matches an optional schema
  - Validates JSON syntax
  - Optional JSON Schema validation
  - Works with both strings and parsed objects
"""

import json

from jsonschema import ValidationError, validate

from autoevals.partial import ScorerWithPartial

from .number import NumericDiff
from .score import Score, Scorer
from .string import Levenshtein


class JSONDiff(ScorerWithPartial):
    """Compare JSON objects for structural and content similarity.

    This scorer recursively compares JSON objects, handling:
    - Nested dictionaries and lists
    - String similarity using Levenshtein distance
    - Numeric value comparison
    - Automatic parsing of JSON strings

    Example:
        ```python
        import asyncio
        from openai import AsyncOpenAI
        from autoevals import JSONDiff
        from autoevals.string import EmbeddingSimilarity

        async def compare_json():
            # Initialize with async client for string comparison
            client = AsyncOpenAI()
            string_scorer = EmbeddingSimilarity(client=client)

            diff = JSONDiff(string_scorer=string_scorer)

            result = await diff.eval_async(
                output={
                    "name": "John Smith",
                    "age": 30,
                    "skills": ["python", "javascript"]
                },
                expected={
                    "name": "John A. Smith",
                    "age": 31,
                    "skills": ["python", "typescript"]
                }
            )

            print(result.score)  # Similarity score between 0-1
            print(result.metadata)  # Detailed comparison breakdown

        # Run the async evaluation
        asyncio.run(compare_json())
        ```

    Args:
        string_scorer: Optional custom scorer for string comparisons (default: Levenshtein)
        number_scorer: Optional custom scorer for number comparisons (default: NumericDiff)
        preserve_strings: Don't attempt to parse strings as JSON (default: False)

    Returns:
        Score object with:
        - score: Similarity score between 0-1
        - metadata: Detailed comparison breakdown
    """

    def __init__(self, string_scorer: Scorer = None, number_scorer: Scorer = None, preserve_strings: bool = False):
        self.string_scorer = string_scorer or Levenshtein()
        self.number_scorer = number_scorer or NumericDiff()
        self.preserve_strings = preserve_strings
        self._valid_json = ValidJSON()

    def _run_eval_sync(self, output, expected=None, **kwargs):
        return Score(name=self._name(), score=self.json_diff(output, expected))

    def json_diff(self, o1, o2):
        if not self.preserve_strings:
            if isinstance(o1, str) and self._valid_json.valid_json(o1) == 1:
                o1 = json.loads(o1)
            if isinstance(o2, str) and self._valid_json.valid_json(o2) == 1:
                o2 = json.loads(o2)

        if isinstance(o1, dict) and isinstance(o2, dict):
            if len(o1) == 0 and len(o2) == 0:
                return 1

            all_keys = set(o1.keys()).union(set(o2.keys()))
            base_scores = [self.json_diff(o1.get(k), o2.get(k)) for k in all_keys]
            base_scores = [s for s in base_scores if s is not None]
            return sum(base_scores) / len(base_scores)
        elif isinstance(o1, list) and isinstance(o2, list):
            if len(o1) == 0 and len(o2) == 0:
                return 1
            base_scores = [self.json_diff(e1, e2) for (e1, e2) in zip(o1, o2)]
            base_scores = [s for s in base_scores if s is not None]
            return sum(base_scores) / max(len(o1), len(o2))
        elif isinstance(o1, str) and isinstance(o2, str):
            return self.string_scorer.eval(o1, o2).score
        elif (isinstance(o1, int) or isinstance(o1, float)) and (isinstance(o2, int) or isinstance(o2, float)):
            return self.number_scorer.eval(o1, o2).score
        elif o1 is None and o2 is None:
            return 1
        elif o1 is None or o2 is None:
            return 0
        else:
            kwargs = {"separators": (",", ":"), "sort_keys": True}
            return self.string_scorer.eval(json.dumps(o1, **kwargs), json.dumps(o2, **kwargs)).score


class ValidJSON(ScorerWithPartial):
    """Validate if a string is valid JSON and optionally matches a schema.

    This scorer checks if:
    - The input can be parsed as valid JSON
    - The parsed JSON matches an optional JSON Schema
    - Handles both string inputs and pre-parsed JSON objects

    Example:
        ```python
        import asyncio
        from autoevals import ValidJSON

        async def validate_json():
            # Define a schema to validate against
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "age"]
            }

            validator = ValidJSON(schema=schema)

            result = await validator.eval_async(
                output='''
                {
                    "name": "John Smith",
                    "age": 30,
                    "skills": ["python", "javascript"]
                }
                '''
            )

            print(result.score)  # 1 if valid, 0 if invalid
            print(result.metadata)  # Validation details or error messages

        # Run the async validation
        asyncio.run(validate_json())
        ```

    Args:
        schema: Optional JSON Schema to validate against

    Returns:
        Score object with:
        - score: 1 if valid JSON (and matches schema if provided), 0 otherwise
        - metadata: Validation details or error messages
    """

    def __init__(self, schema=None):
        self.schema = schema

    def _run_eval_sync(self, output, schema=None, **kwargs):
        return Score(name=self._name(), score=self.valid_json(output, schema))

    def valid_json(self, output, schema=None):
        try:
            parsed = json.loads(output) if isinstance(output, str) else output

            if schema is not None:
                validate(parsed, schema)
                return 1

            if isinstance(parsed, dict) or isinstance(parsed, list):
                return 1

        except (json.JSONDecodeError, ValidationError):
            pass

        return 0


__all__ = ["JSONDiff", "ValidJSON"]
