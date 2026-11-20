import asyncio
import json

import pytest
import respx
from httpx import Response
from openai import AsyncOpenAI, OpenAI

import autoevals.string as string_module
from autoevals import EmbeddingSimilarity, init
from autoevals.oai import LLMClient

SYNONYMS = [
    ("water", ["water", "H2O", "agua"]),
    ("fire", ["fire", "flame"]),
    ("earth", ["earth", "Planet Earth"]),
]

UNRELATED = ["water", "The quick brown fox jumps over the lazy dog", "I like to eat apples"]


@pytest.fixture(autouse=True)
def reset_embedding_cache():
    with EmbeddingSimilarity._CACHE_LOCK:
        EmbeddingSimilarity._CACHE.clear()
    yield
    with EmbeddingSimilarity._CACHE_LOCK:
        EmbeddingSimilarity._CACHE.clear()


@pytest.fixture
def embedding_providers():
    def handle(request):
        body = json.loads(request.content)
        vector = (
            [0.0, 1.0] if request.url.host == "provider-b.example" and body["input"] == "reference" else [1.0, 0.0]
        )
        return Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": [{"object": "embedding", "index": 0, "embedding": vector}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    init()
    with respx.mock as router:
        routes = [
            router.post(f"https://provider-{provider}.example/v1/embeddings").mock(side_effect=handle)
            for provider in ("a", "b")
        ]
        yield routes
    init()


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.asyncio
async def test_embedding_cache_isolated_by_default_client(embedding_providers, run_async):
    client_type = AsyncOpenAI if run_async else OpenAI
    clients = [
        client_type(api_key="test-key", base_url=f"https://provider-{provider}.example/v1") for provider in ("a", "b")
    ]
    wrappers = [LLMClient(openai=client) for client in clients]
    evaluator = EmbeddingSimilarity(expected_min=0)

    async def score(metric):
        if run_async:
            return (await metric.eval_async("output", "reference")).score
        return (await asyncio.to_thread(metric.eval, "output", "reference")).score

    try:
        init(client=wrappers[0])
        assert await score(evaluator) == 1.0
        assert await score(EmbeddingSimilarity(expected_min=0)) == 1.0
        assert embedding_providers[0].call_count == 2

        init(client=wrappers[1])
        assert await score(evaluator) == 0.0
        assert await score(EmbeddingSimilarity(expected_min=0)) == 0.0
        assert embedding_providers[1].call_count == 2

        assert await score(EmbeddingSimilarity(client=wrappers[0], expected_min=0)) == 1.0
        init(client=wrappers[0])
        assert await score(evaluator) == 1.0
        assert embedding_providers[0].call_count == 2
    finally:
        init()
        for client in clients:
            if run_async:
                await client.close()
            else:
                client.close()


@pytest.mark.asyncio
async def test_embedding_cache_isolated_by_async_client_context(embedding_providers):
    async with (
        AsyncOpenAI(api_key="test-key", base_url="https://provider-a.example/v1") as client_a,
        AsyncOpenAI(api_key="test-key", base_url="https://provider-b.example/v1") as client_b,
    ):
        ready = [asyncio.Event(), asyncio.Event()]
        evaluator = EmbeddingSimilarity(expected_min=0)

        async def evaluate(client, index):
            init(client=client)
            ready[index].set()
            await ready[1 - index].wait()
            first = await evaluator.eval_async("output", "reference")
            second = await evaluator.eval_async("output", "reference")
            return first.score, second.score

        results = await asyncio.gather(evaluate(client_a, 0), evaluate(client_b, 1))

    assert results == [(1.0, 1.0), (0.0, 0.0)]
    assert [route.call_count for route in embedding_providers] == [2, 2]


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.asyncio
async def test_embedding_cache_isolated_by_configuration(monkeypatch, run_async):
    calls = []

    def fake_run_cached_request(**kwargs):
        calls.append((kwargs["model"], kwargs["input"], kwargs.get("base_url"), kwargs.get("client")))
        return {"data": [{"embedding": [1.0, 0.0]}]}

    async def fake_arun_cached_request(**kwargs):
        calls.append((kwargs["model"], kwargs["input"], kwargs.get("base_url"), kwargs.get("client")))
        return {"data": [{"embedding": [1.0, 0.0]}]}

    if run_async:
        monkeypatch.setattr(string_module, "arun_cached_request", fake_arun_cached_request)
    else:
        monkeypatch.setattr(string_module, "run_cached_request", fake_run_cached_request)

    value = {"topic": "cache"}
    client_a = object()
    client_b = object()
    evaluators = [
        EmbeddingSimilarity(model="model-a", prefix="query: ", base_url="https://one.example", client=client_a),
        EmbeddingSimilarity(model="model-a", prefix="query: ", base_url="https://one.example", client=client_a),
        EmbeddingSimilarity(model="model-b", prefix="query: ", base_url="https://one.example", client=client_a),
        EmbeddingSimilarity(model="model-a", prefix="document: ", base_url="https://one.example", client=client_a),
        EmbeddingSimilarity(model="model-a", prefix="query: ", base_url="https://two.example", client=client_a),
        EmbeddingSimilarity(model="model-a", prefix="query: ", base_url="https://one.example", client=client_b),
    ]

    for evaluator in evaluators:
        if run_async:
            await evaluator.eval_async(value, value)
        else:
            evaluator.eval(value, value)

    assert calls == [
        ("model-a", 'query: {"topic": "cache"}', "https://one.example", client_a),
        ("model-b", 'query: {"topic": "cache"}', "https://one.example", client_a),
        ("model-a", 'document: {"topic": "cache"}', "https://one.example", client_a),
        ("model-a", 'query: {"topic": "cache"}', "https://two.example", client_a),
        ("model-a", 'query: {"topic": "cache"}', "https://one.example", client_b),
    ]


def test_embeddings():
    evaluator = EmbeddingSimilarity(prefix="resource type: ")
    for word, synonyms in SYNONYMS:
        for synonym in synonyms:
            result = evaluator(word, synonym)
            print(f"[{word}]", f"[{synonym}]", result)
            assert result.score > 0.66

    for i in range(len(UNRELATED)):
        for j in range(len(UNRELATED)):
            if i == j:
                continue

            word1 = UNRELATED[i]
            word2 = UNRELATED[j]
            result = evaluator(word1, word2)
            print(f"[{word1}]", f"[{word2}]", result)
            assert result.score < 0.5


VALUES = [
    ("water", "wind"),
    (["cold", "water"], ["cold", "wind"]),
    ({"water": "wet"}, {"wind": "dry"}),
]


def test_embedding_values():
    for run_async in [False, True]:
        evaluator = EmbeddingSimilarity()
        for word1, word2 in VALUES:
            if run_async:
                result = asyncio.run(evaluator.eval_async(word1, word2))
            else:
                result = evaluator(word1, word2)
            print(f"[{word1}]", f"[{word2}]", f"run_async={run_async}", result)
