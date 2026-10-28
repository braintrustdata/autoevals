import json

import httpx
import pytest
from openai import AsyncOpenAI

from autoevals.json import JSONDiff
from autoevals.score import Score, Scorer
from autoevals.string import EmbeddingSimilarity


class AsyncOnlyScorer(Scorer):
    def _run_eval_sync(self, output, expected=None, **kwargs):
        raise AssertionError("Synchronous scorer called during async evaluation")

    async def _run_eval_async(self, output, expected=None, **kwargs):
        return Score(name=self._name(), score=0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output,expected",
    [
        ({"answer": ["Paris"]}, {"answer": ["City of Paris"]}),
        ('{"answer": ["Paris"]}', '{"answer": ["City of Paris"]}'),
        ({"answer": [1]}, {"answer": [2]}),
        ([{}], [[]]),
    ],
)
async def test_json_diff_awaits_custom_scorers(output, expected):
    scorer = JSONDiff(string_scorer=AsyncOnlyScorer(), number_scorer=AsyncOnlyScorer())
    assert (await scorer.eval_async(output, expected)).score == 0.5


@pytest.mark.asyncio
async def test_json_diff_partial_awaits_custom_scorer():
    scorer = JSONDiff.partial(expected={"answer": "Paris"})(string_scorer=AsyncOnlyScorer())
    assert (await scorer.eval_async({"answer": "City of Paris"})).score == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output,expected,preserve_strings",
    [
        ({}, {}, False),
        ([], [], False),
        (None, None, False),
        ({"answer": None}, {"answer": "Paris"}, False),
        ({"a": [1, "Paris"]}, {"a": [2, "Pari"], "extra": 1}, False),
        ([1, 2], [1], False),
        ('{"a": 1}', '{"a": 2}', False),
        ('{"a": 1}', '{"a": 2}', True),
        (True, False, False),
    ],
)
async def test_json_diff_async_preserves_sync_scores(output, expected, preserve_strings):
    scorer = JSONDiff(preserve_strings=preserve_strings)
    assert (await scorer.eval_async(output, expected)).score == scorer.eval(output, expected).score


@pytest.mark.asyncio
async def test_json_diff_with_async_embedding_client():
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body["model"], body["input"]))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
                "model": "test-embedding",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    with EmbeddingSimilarity._CACHE_LOCK:
        EmbeddingSimilarity._CACHE.clear()
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
            async with AsyncOpenAI(
                api_key="test-key", base_url="https://embeddings.example/v1", http_client=http_client
            ) as client:
                scorer = JSONDiff(string_scorer=EmbeddingSimilarity(client=client, model="test-embedding"))
                result = await scorer.eval_async({"answer": ["Paris"]}, {"answer": ["City of Paris"]})
        assert result.score == 1
        assert requests == [
            ("/v1/embeddings", "test-embedding", "Paris"),
            ("/v1/embeddings", "test-embedding", "City of Paris"),
        ]
    finally:
        with EmbeddingSimilarity._CACHE_LOCK:
            EmbeddingSimilarity._CACHE.clear()
