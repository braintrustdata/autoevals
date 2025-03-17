from typing import Optional

from braintrust_core.score import Score

from autoevals.llm import OpenAIScorer

from .oai import Client, arun_cached_request, run_cached_request

REQUEST_TYPE = "moderation"


class Moderation(OpenAIScorer):
    """A scorer that evaluates if AI responses contain inappropriate or unsafe content.

    This scorer uses OpenAI's moderation API to check for potentially harmful content in text.
    It returns a score of 0 if content is flagged as inappropriate, and 1 if it passes moderation.

    Args:
        threshold: Optional float to set a custom threshold for content flagging.
            If None, uses OpenAI's default flagging logic.
            If set, content is flagged if any category score exceeds this threshold.
        client: Optional client for making API calls. Can be:
            - None: Uses global client from init()
            - OpenAI v1 client: Used directly
            - OpenAI v0 module: Wrapped in a client adapter

    Example:
        ```python
        from openai import OpenAI
        from autoevals import init
        from autoevals.moderation import Moderation

        # Initialize with your OpenAI client
        init(OpenAI())

        # Create evaluator with default settings
        moderator = Moderation()
        result = moderator.eval(
            output="This is the text to check for inappropriate content"
        )
        print(result.score)  # 1 if content is appropriate, 0 if flagged
        print(result.metadata)  # Detailed category scores and threshold used
        ```
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
            threshold: Optional float to set a custom threshold for content flagging.
                If None, uses OpenAI's default flagging logic.
                If set, content is flagged if any category score exceeds this threshold.
            client: Optional client for making API calls. Can be:
                - None: Uses global client from init()
                - OpenAI v1 client: Used directly
                - OpenAI v0 module: Wrapped in a client adapter
            api_key: Deprecated. Use client instead.
            base_url: Deprecated. Use client instead.

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
