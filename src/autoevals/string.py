from Levenshtein import distance

from .base import Evaluator


class LevenshteinEvaluator(Evaluator):
    async def _run_eval_async(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("LevenshteinEvaluator requires an expected value")

        output, expected = str(output), str(expected)
        max_len = max(len(x) for x in [output, expected])
        if max_len == 0:
            return 1

        return 1 - (distance(output, expected) / max(len(x) for x in [output, expected]))


__all__ = ["LevenshteinEvaluator"]
