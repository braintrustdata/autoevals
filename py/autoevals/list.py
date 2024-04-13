import sys

from braintrust_core.score import Score, Scorer

from .string import Levenshtein


class ListContains(Scorer):
    """
    A scorer that semantically evaluates the overlap between two lists of strings. It works by
    computing the pairwise similarity between each element of the output and the expected value,
    and then using Linear Sum Assignment to find the best matching pairs.
    """

    def __init__(self, pairwise_scorer=None, allow_extra_entities=False, **kwargs):
        self.allow_extra_entities = allow_extra_entities
        self.pairwise_scorer = pairwise_scorer or Levenshtein()

    async def _run_eval_async(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("ListContains requires an expected value")

        similarities_futures = [
            [self.pairwise_scorer._run_eval_async(output_item, expected_item) for expected_item in expected]
            for output_item in output
        ]

        similarities = []

        for similarity_futures in similarities_futures:
            similarities.append([(await similarity_future).score for similarity_future in similarity_futures])

        return self._compute_score(output, expected, similarities, **kwargs)

    def _run_eval_sync(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("ListContains requires an expected value")

        similarities = [
            [self.pairwise_scorer._run_eval_sync(output_item, expected_item).score for expected_item in expected]
            for output_item in output
        ]

        return self._compute_score(output, expected, similarities, **kwargs)

    def _compute_score(self, outputs, expecteds, similarities, **kwargs):
        if len(outputs) == 0 and len(expecteds) == 0:
            return Score(name=self._name(), score=1)
        elif len(outputs) == 0 or len(expecteds) == 0:
            return Score(name=self._name(), score=0)

        similarities = [[d or 0 for d in row] for row in similarities]

        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            print(
                "ListContains requires the scipy extension, which you can install with `pip install 'autoevals[scipy]'`",
                file=sys.stderr,
            )
            raise

        distances = -np.array(similarities)
        row_ind, col_ind = linear_sum_assignment(distances)

        pairs = [(outputs[r], expecteds[c], similarities[r][c]) for (r, c) in zip(row_ind, col_ind)]
        lowest_distances = distances[row_ind, col_ind]

        # Generally speaking, outputs that are not in expecteds should be penalized, but in certain use cases
        # (eg checking whether a passage of text has all of the entities in a list, and maybe a few more), it's
        # ok to allow them.
        denominator = max(len(outputs), len(expecteds)) if not self.allow_extra_entities else len(expecteds)
        assert len(lowest_distances) <= denominator, "There should be at most as many pairs as there are rows"
        score = min(max(sum(-lowest_distances) / denominator, 0), 1)

        return Score(
            name=self._name(),
            score=score,
            metadata={"pairs": pairs},
        )
