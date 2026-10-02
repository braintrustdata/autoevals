import asyncio

import pytest

import autoevals.string as string_module
from autoevals import EmbeddingSimilarity

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


def test_embedding_cache_isolated_by_model_and_prefix(monkeypatch):
    calls = []

    def fake_run_cached_request(**kwargs):
        calls.append((kwargs["model"], kwargs["input"]))
        return {"data": [{"embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(string_module, "run_cached_request", fake_run_cached_request)
    value = {"topic": "cache"}

    EmbeddingSimilarity(model="model-a", prefix="query: ").eval(value, value)
    EmbeddingSimilarity(model="model-a", prefix="query: ").eval(value, value)
    EmbeddingSimilarity(model="model-b", prefix="query: ").eval(value, value)
    EmbeddingSimilarity(model="model-a", prefix="document: ").eval(value, value)

    assert calls == [
        ("model-a", 'query: {"topic": "cache"}'),
        ("model-b", 'query: {"topic": "cache"}'),
        ("model-a", 'document: {"topic": "cache"}'),
    ]


@pytest.mark.asyncio
async def test_async_embedding_cache_isolated_by_model_and_prefix(monkeypatch):
    calls = []

    async def fake_arun_cached_request(**kwargs):
        calls.append((kwargs["model"], kwargs["input"]))
        return {"data": [{"embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(string_module, "arun_cached_request", fake_arun_cached_request)
    value = {"topic": "cache"}

    await EmbeddingSimilarity(model="model-a", prefix="query: ").eval_async(value, value)
    await EmbeddingSimilarity(model="model-a", prefix="query: ").eval_async(value, value)
    await EmbeddingSimilarity(model="model-b", prefix="query: ").eval_async(value, value)
    await EmbeddingSimilarity(model="model-a", prefix="document: ").eval_async(value, value)

    assert calls == [
        ("model-a", 'query: {"topic": "cache"}'),
        ("model-b", 'query: {"topic": "cache"}'),
        ("model-a", 'document: {"topic": "cache"}'),
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
