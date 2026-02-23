"""String evaluation scorers for comparing text similarity.

This module provides scorers for text comparison:

- Levenshtein: Compare strings using edit distance
  - Fast, local string comparison
  - Suitable for exact matches and small variations
  - No external dependencies
  - Simple to use with just output/expected parameters

- EmbeddingSimilarity: Compare strings using embeddings
  - Semantic similarity using embeddings
  - Requires OpenAI API access
  - Better for comparing meaning rather than exact matches
  - Supports both sync and async evaluation
  - Built-in caching for efficiency
  - Configurable with options for model, prefix, thresholds
"""

import threading

from polyleven import levenshtein as distance

from autoevals.partial import ScorerWithPartial
from autoevals.value import normalize_value

from .oai import LLMClient, arun_cached_request, get_default_embedding_model, run_cached_request
from .score import Score


class Levenshtein(ScorerWithPartial):
    """String similarity scorer using edit distance.

    Example:
        ```python
        scorer = Levenshtein()
        result = scorer.eval(
            output="hello wrld",
            expected="hello world"
        )
        print(result.score)  # 0.9 (normalized similarity)
        ```

    Args:
        output: String to evaluate
        expected: Reference string to compare against

    Returns:
        Score object with normalized similarity (0-1), where 1 means identical strings
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


class EmbeddingSimilarity(ScorerWithPartial):
    """String similarity scorer using embeddings.

    Example:
        ```python
        import asyncio
        from openai import AsyncOpenAI
        from autoevals.string import EmbeddingSimilarity

        async def compare_texts():
            # Initialize with async client
            client = AsyncOpenAI()
            scorer = EmbeddingSimilarity(
                prefix="Code explanation: ",
                client=client
            )

            result = await scorer.eval_async(
                output="The function sorts elements using quicksort",
                expected="The function implements quicksort algorithm"
            )

            print(result.score)  # 0.85 (normalized similarity)
            print(result.metadata)  # Additional comparison details

        # Run the async evaluation
        asyncio.run(compare_texts())
        ```

    Args:
        prefix: Optional text to prepend to inputs for domain context
        model: Embedding model to use (default: text-embedding-ada-002)
        expected_min: Minimum similarity threshold (default: 0.7)
        client: Optional AsyncOpenAI/OpenAI client. If not provided, uses global client from init()

    Returns:
        Score object with:
        - score: Normalized similarity (0-1)
        - metadata: Additional comparison details
    """

    MODEL = "text-embedding-ada-002"

    _CACHE = {}
    _CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        prefix="",
        model=None,
        expected_min=0.7,
        api_key=None,
        base_url=None,
        client: LLMClient | None = None,
    ):
        self.prefix = prefix
        self.expected_min = expected_min

        self.extra_args = {"model": model if model is not None else get_default_embedding_model()}
        if api_key:
            self.extra_args["api_key"] = api_key
        if base_url:
            self.extra_args["base_url"] = base_url

        self.client = client

    async def _a_embed(self, value):
        value = normalize_value(value, maybe_object=False)
        with self._CACHE_LOCK:
            if value in self._CACHE:
                return self._CACHE[value]

        result = await arun_cached_request(
            client=self.client, request_type="embed", input=f"{self.prefix}{value}", **self.extra_args
        )

        with self._CACHE_LOCK:
            self._CACHE[value] = result

        return result

    def _embed(self, value):
        value = normalize_value(value, maybe_object=False)
        with self._CACHE_LOCK:
            if value in self._CACHE:
                return self._CACHE[value]

        result = run_cached_request(
            client=self.client, request_type="embed", input=f"{self.prefix}{value}", **self.extra_args
        )

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
