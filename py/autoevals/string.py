import threading

from braintrust_core.score import Score, Scorer
from Levenshtein import distance

from .oai import arun_cached_request, run_cached_request


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


class EmbeddingSimilarity(Scorer):
    """
    A simple scorer that uses cosine similarity to compare two strings.
    """

    MODEL = "text-embedding-ada-002"

    _CACHE = {}
    _CACHE_LOCK = threading.Lock()

    def __init__(self, prefix="", model=MODEL, expected_min=0.7, api_key=None, base_url=None):
        """
        Create a new EmbeddingSimilarity scorer.

        :param prefix: A prefix to prepend to the prompt. This is useful for specifying the domain of the inputs.
        :param model: The model to use for the embedding distance. Defaults to "text-embedding-ada-002".
        :param expected_min: The minimum expected score. Defaults to 0.7. Values below this will be scored as 0, and
        values between this and 1 will be scaled linearly.
        """
        self.prefix = prefix
        self.expected_min = expected_min

        self.extra_args = {"model": model}
        if api_key:
            self.extra_args["api_key"] = api_key
        if base_url:
            self.extra_args["base_url"] = base_url

    async def _a_embed(self, value):
        with self._CACHE_LOCK:
            if value in self._CACHE:
                return self._CACHE[value]

        result = await arun_cached_request("embed", input=f"{self.prefix}{value}", **self.extra_args)

        with self._CACHE_LOCK:
            self._CACHE[value] = result

        return result

    def _embed(self, value):
        with self._CACHE_LOCK:
            if value in self._CACHE:
                return self._CACHE[value]

        result = run_cached_request("embed", input=f"{self.prefix}{value}", **self.extra_args)

        with self._CACHE_LOCK:
            self._CACHE[value] = result

        return result

    async def _run_eval_async(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("EmbeddingSimilarity requires an expected value")

        output_embedding_p = self._a_embed(output)
        expected_embedding_p = self._a_embed(expected)

        output_result, expected_result = await output_embedding_p, await expected_embedding_p
        return Score(
            name=self._name(),
            score=self.scale_score(
                self.cosine_similarity(output_result["data"][0]["embedding"], expected_result["data"][0]["embedding"]),
                self.expected_min,
            ),
        )

    def _run_eval_sync(self, output, expected=None, **kwargs):
        if expected is None:
            raise ValueError("EmbeddingSimilarity requires an expected value")

        output_result = self._embed(output)
        expected_result = self._embed(expected)

        return Score(
            name=self._name(),
            score=self.scale_score(
                self.cosine_similarity(output_result["data"][0]["embedding"], expected_result["data"][0]["embedding"]),
                self.expected_min,
            ),
        )

    @staticmethod
    def scale_score(score, expected_min):
        return max((score - expected_min) / (1 - expected_min), 0)

    @staticmethod
    def cosine_similarity(list1, list2):
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(list1, list2))

        # Calculate the magnitude of each list
        magnitude_list1 = sum(a**2 for a in list1) ** 0.5
        magnitude_list2 = sum(b**2 for b in list2) ** 0.5

        # Calculate cosine similarity
        if magnitude_list1 * magnitude_list2 == 0:
            # Avoid division by zero
            return 0
        else:
            # Sometimes, rounding errors cause the dot product to be slightly > 1
            return min(dot_product / (magnitude_list1 * magnitude_list2), 1)


__all__ = ["LevenshteinScorer", "Levenshtein", "EmbeddingSimilarity"]
