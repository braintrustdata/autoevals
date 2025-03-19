"""Numeric evaluation scorers for comparing numerical values.

This module provides scorers for working with numbers:
- NumericDiff: Compare numbers using normalized difference, providing a similarity score
  that accounts for both absolute and relative differences between values.

Features:
- Normalized scoring between 0 and 1
- Handles special cases like comparing zeros
- Accounts for magnitude when computing differences
- Suitable for both small and large number comparisons
"""

from braintrust_core.score import Score

from autoevals.partial import ScorerWithPartial


class NumericDiff(ScorerWithPartial):
    """Numeric similarity scorer using normalized difference.

    Example:
        ```python
        scorer = NumericDiff()
        result = scorer.eval(
            output=105,
            expected=100
        )
        print(result.score)  # 0.95 (normalized similarity)
        ```

    Args:
        output: Number to evaluate
        expected: Reference number to compare against

    Returns:
        Score object with normalized similarity (0-1), where:
        - 1 means identical numbers
        - Score decreases as difference increases relative to magnitude
        - Special case: score=1 when both numbers are 0
    """

    def _run_eval_sync(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("NumericDiff requires an expected value")

        if expected == 0 and output == 0:
            score = 1
        else:
            score = 1 - abs(expected - output) / (abs(expected) + abs(output))
        return Score(name=self._name(), score=score)


__all__ = ["NumericDiff"]
