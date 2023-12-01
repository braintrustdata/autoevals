from braintrust_core.score import Score, Scorer


class NumericDiff(Scorer):
    """
    A simple scorer that compares numbers by normalizing their difference.
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
