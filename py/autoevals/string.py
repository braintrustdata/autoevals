from Levenshtein import distance

from autoevals.base import Score

from .base import Score, Scorer


class Levenshtein(Scorer):
    """
    A simple scorer that uses the Levenshtein distance to compare two strings.
    """

    def _run_eval_sync(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("LevenshteinScorer requires an expected value")

        output, expected = str(output), str(expected)
        max_len = max(len(x) for x in [output, expected])

        score = 1
        if max_len > 0:
            score = 1 - (distance(output, expected) / max_len)

        return Score(name=self._name(), score=score)


LevenshteinScorer = Levenshtein  # backcompat


__all__ = ["LevenshteinScorer", "Levenshtein"]
