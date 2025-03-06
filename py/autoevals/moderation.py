from typing import Optional

from braintrust_core.score import Score

from autoevals.llm import OpenAIScorer

from .oai import Client, arun_cached_request, run_cached_request

REQUEST_TYPE = "moderation"


class Moderation(OpenAIScorer):
    """A scorer that uses OpenAI's moderation API to determine if AI response contains ANY flagged content.

    Args:
        threshold: Optional threshold to use to determine whether content has exceeded threshold.
            By default, uses OpenAI's default (using `flagged` from the response payload).
        api_key: Deprecated. Use client instead.
        base_url: Deprecated. Use client instead.
        client: Optional Client. If not provided, uses global client from init().

    Note:
        The api_key and base_url parameters are deprecated and will be removed in a future version.
        Instead, you can either:
        1. Pass a client instance directly to this constructor using the client parameter
        2. Set a global client using autoevals.init(client=your_client)

        The global client can be configured once and will be used by all evaluators that don't have
        a specific client passed to them.
    """

    threshold = None
    extra_args = {}

    def __init__(
        self,
        threshold=None,
        api_key=None,
        base_url=None,
        client: Optional[Client] = None,
    ):
        """Initialize a Moderation scorer.

        Args:
            threshold: Optional threshold to use to determine whether content has exceeded threshold.
                By default, uses OpenAI's default (using `flagged` from the response payload).
            api_key: Deprecated. Use client instead.
            base_url: Deprecated. Use client instead.
            client: Optional Client. If not provided, uses global client from init().

        Note:
            The api_key and base_url parameters are deprecated and will be removed in a future version.
            Instead, you can either:
            1. Pass a client instance directly to this constructor using the client parameter
            2. Set a global client using autoevals.init(client=your_client)

            The global client can be configured once and will be used by all evaluators that don't have
            a specific client passed to them.
        """
        super().__init__(api_key=api_key, base_url=base_url, client=client)
        self.threshold = threshold

    def _run_eval_sync(self, output, expected=None, **kwargs):
        moderation_response = run_cached_request(
            client=self.client, request_type=REQUEST_TYPE, input=output, **self.extra_args
        )["results"][0]
        return self.__postprocess_response(moderation_response)

    def __postprocess_response(self, moderation_response) -> Score:
        return Score(
            name=self._name(),
            score=self.compute_score(moderation_response, self.threshold),
            metadata={
                "threshold": self.threshold,
                "category_scores": moderation_response["category_scores"],
            },
        )

    async def _run_eval_async(self, output, expected=None, **kwargs) -> Score:
        moderation_response = (
            await arun_cached_request(client=self.client, request_type=REQUEST_TYPE, input=output, **self.extra_args)
        )["results"][0]
        return self.__postprocess_response(moderation_response)

    @staticmethod
    def compute_score(moderation_result, threshold):
        if threshold is None:
            return 0 if moderation_result["flagged"] else 1

        category_scores = moderation_result["category_scores"]
        for category in category_scores.keys():
            if category_scores[category] > threshold:
                return 0

        return 1


__all__ = ["Moderation"]
