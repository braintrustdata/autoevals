from braintrust_core.score import Score, Scorer

from autoevals import Levenshtein


class ListContains(Scorer):
    def __init__(self, pairwise_scorer=None, allow_extra_entities=False, **kwargs):
        self.allow_extra_entities = allow_extra_entities
        self.pairwise_scorer = pairwise_scorer or Levenshtein()

    async def _run_eval_async(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("ListContains requires an expected value")

        distances_futures = [
            [self.pairwise_scorer._run_eval_async(output_item, expected_item) for expected_item in expected]
            for output_item in output
        ]

        distances = []

        for distance_futures in distances_futures:
            distances.append([(await distance_future).score for distance_future in distance_futures])

        return self._compute_scores(output, expected, distances, **kwargs)

    def _run_eval_sync(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("ListOverlap requires an expected value")

        distances = [
            [self.pairwise_scorer._run_eval_sync(output_item, expected_item).score or 0 for expected_item in expected]
            for output_item in output
        ]

        return self._compute_scores(output, expected, distances, **kwargs)

    def _compute_scores(self, outputs, expecteds, distances, **kwargs):
        if len(outputs) == 0 and len(expecteds) == 0:
            return Score(name=self._name(), score=1)
        elif len(outputs) == 0 or len(expecteds) == 0:
            return Score(name=self._name(), score=0)

        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            print(
                "ListOverlap requires the scipy extension, which you can install with `pip install 'autoevals[scipy]'`",
                file=sys.stderr,
            )
            raise

        distances = 1 - np.array(distances)
        row_ind, col_ind = linear_sum_assignment(distances)

        pairs = [(outputs[r], expecteds[c], 1 - distances[r][c]) for (r, c) in zip(row_ind, col_ind)]
        lowest_distances = distances[row_ind, col_ind]

        # Generally speaking, outputs that are not in expecteds should be penalized, but in certain use cases
        # (eg checking whether a passage of text has all of the entities in a list, and maybe a few more), it's
        # ok to allow them.
        denominator = max(len(outputs), len(expecteds)) if not self.allow_extra_entities else len(expecteds)
        assert len(lowest_distances) <= denominator, "There should be at most as many pairs as there are rows"
        score = sum(1 - lowest_distances) / denominator

        return Score(
            name=self._name(),
            score=score,
            metadata={"pairs": pairs, "lowest_distances": lowest_distances.tolist()},
        )
