from Levenshtein import distance

from autoevals.base import Evaluation

from .base import Evaluation, Evaluator


class LevenshteinEvaluator(Evaluator):
    def _run_eval_sync(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("LevenshteinEvaluator requires an expected value")

        output, expected = str(output), str(expected)
        max_len = max(len(x) for x in [output, expected])

        if max_len == 0:
            score = 1
        else:
            score = 1 - (distance(output, expected) / max(len(x) for x in [output, expected]))

        return Evaluation(score=score)


__all__ = ["LevenshteinEvaluator"]
