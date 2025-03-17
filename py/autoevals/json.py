import json

from braintrust_core.score import Score, Scorer
from jsonschema import ValidationError, validate

from autoevals.partial import ScorerWithPartial

from .number import NumericDiff
from .string import Levenshtein


class JSONDiff(ScorerWithPartial):
    """
    A simple scorer that compares JSON objects, using a customizable comparison method for strings
    (defaults to Levenshtein) and numbers (defaults to NumericDiff).
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
    """
    A binary scorer that evaluates the validity of JSON output, optionally validating against a
    JSON Schema definition (see https://json-schema.org/learn/getting-started-step-by-step#create).
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
