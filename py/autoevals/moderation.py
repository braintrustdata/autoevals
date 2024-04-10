import threading

from braintrust_core.score import Score, Scorer

from .oai import arun_cached_request, run_cached_request

REQUEST_TYPE = "moderation"


class Moderation(Scorer):
    """
    A scorer that uses OpenAI's moderation API to determine if AI response contains ANY flagged content.
    """

    _CACHE = {}
    _CACHE_LOCK = threading.Lock()

    threshold = None
    extra_args = {}

    def __init__(self, threshold=None, api_key=None, base_url=None):
        """
        Create a new Moderation scorer.

        :param threshold: A prefix to prepend to the prompt. This is useful for specifying the domain of the inputs.
        :param api_key: OpenAI key
        :param base_url: Base URL to be used to reach OpenAI moderation endpoint.
        """
        self.threshold = threshold

        self.extra_args = {}

        if api_key:
            self.extra_args["api_key"] = api_key
        if base_url:
            self.extra_args["base_url"] = base_url

    async def _a_moderation(self, value):
        with self._CACHE_LOCK:
            if value in self._CACHE:
                return self._CACHE[value]

        result = await arun_cached_request(REQUEST_TYPE, input=value, **self.extra_args)

        with self._CACHE_LOCK:
            self._CACHE[value] = result

        return result

    def _moderation(self, value):
        with self._CACHE_LOCK:
            if value in self._CACHE:
                return self._CACHE[value]

        result = run_cached_request(REQUEST_TYPE, input=value, **self.extra_args)

        with self._CACHE_LOCK:
            self._CACHE[value] = result

        return result

    async def _run_eval_async(self, output, __expected=None):
        output_result = (await self._a_moderation(output))["results"][0]
        return Score(
            name=self._name(),
            score=self.compute_score(output_result, self.threshold),
            metadata=output_result["category_scores"],
        )

    def _run_eval_sync(self, output, __expected=None):
        output_result = self._moderation(output)["results"][0]
        return Score(
            name=self._name(),
            score=self.compute_score(output_result, self.threshold),
            metadata=output_result["category_scores"],
        )

    @staticmethod
    def compute_score(moderation_result, threshold):
        if threshold is None:
            return 0 if moderation_result["flagged"] else 1

        print(moderation_result)

        category_scores = moderation_result["category_scores"]

        for category in category_scores.keys():
            if category_scores[category] > threshold:
                return 0

        return 1


__all__ = ["Moderation"]
