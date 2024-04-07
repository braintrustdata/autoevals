from braintrust_core.score import Score, Scorer

from autoevals import Levenshtein


class ListContains(Scorer):
    def __init__(self, pairwise_scorer=None, **kwargs):
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

    def _compute_scores(self, rows, columns, distances, **kwargs):
        if len(rows) == 0 and len(columns) == 0:
            return Score(name=self._name(), score=1)
        elif len(rows) == 0 or len(columns) == 0:
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

        pairs = [(rows[r], columns[c], 1 - distances[r][c]) for (r, c) in zip(row_ind, col_ind)]
        lowest_distances = distances[row_ind, col_ind]

        # We care about each element of output being _in_ expected, so we also need to penalize
        # elements in expected that are not in output
        assert len(lowest_distances) <= len(columns), "There should be at most as many pairs as there are rows"
        score = sum(1 - lowest_distances) / len(columns)

        return Score(
            name=self._name(),
            score=score,
            metadata={"pairs": pairs, "lowest_distances": lowest_distances.tolist()},
        )
