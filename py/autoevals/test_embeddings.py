import asyncio

from autoevals import EmbeddingSimilarity
from autoevals.value import normalize_value

SYNONYMS = [
    ("water", ["water", "H2O", "agua"]),
    ("fire", ["fire", "flame"]),
    ("earth", ["earth", "Planet Earth"]),
]

UNRELATED = ["water", "The quick brown fox jumps over the lazy dog", "I like to eat apples"]


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
        for (word1, word2) in VALUES:
            if run_async:
                result = asyncio.run(evaluator.eval_async(word1, word2))
            else:
                result = evaluator(word1, word2)
            print(f"[{word1}]", f"[{word2}]", f"run_async={run_async}", result)
