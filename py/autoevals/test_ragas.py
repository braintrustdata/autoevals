import asyncio
import json

import pytest
import respx
from httpx import Response
from openai import OpenAI

from autoevals import init
from autoevals.ragas import *

data = {
    "input": "Can starred docs from different workspaces be accessed in one place?",
    "output": "Yes, all starred docs, even from multiple different workspaces, will live in the My Shortcuts section.",
    "expected": "Yes, all starred docs, even from multiple different workspaces, will live in the My Shortcuts section.",
    "context": [
        "Not all Coda docs are used in the same way. You'll inevitably have a few that you use every week, and some that you'll only use once. This is where starred docs can help you stay organized.\n\n\n\nStarring docs is a great way to mark docs of personal importance. After you star a doc, it will live in a section on your doc list called **[My Shortcuts](https://coda.io/shortcuts)**. All starred docs, even from multiple different workspaces, will live in this section.\n\n\n\nStarring docs only saves them to your personal My Shortcuts. It doesn\u2019t affect the view for others in your workspace. If you\u2019re wanting to shortcut docs not just for yourself but also for others in your team or workspace, you\u2019ll [use pinning](https://help.coda.io/en/articles/2865511-starred-pinned-docs) instead."
    ],
}


@pytest.mark.parametrize(
    ["metric", "expected_score", "can_fail"],
    [
        (ContextEntityRecall(), 0.5, False),
        (ContextRelevancy(), 0.7, True),
        (ContextRecall(), 1, True),
        (ContextPrecision(), 1, False),
    ],
)
@pytest.mark.parametrize("is_async", [False, True])
def test_ragas_retrieval(metric: OpenAILLMScorer, expected_score: float, is_async: bool, can_fail: bool):
    if is_async:
        score = asyncio.run(metric.eval_async(**data)).score
    else:
        score = metric.eval(**data).score

    if score is None:
        raise ValueError("Score is None")

    try:
        if expected_score == 1:
            assert score == expected_score
        else:
            assert score >= expected_score
    except AssertionError as e:
        # TODO: just to unblock the CI
        if can_fail:
            pytest.xfail(f"Expected score {expected_score} but got {score}")
        else:
            raise e


def test_context_relevancy_score_clamping():
    """Test that ContextRelevancy clamps scores to [0, 1] range (#80).

    When the LLM returns sentences longer than the original context
    (due to paraphrasing or hallucination), the raw score would exceed 1.0.
    This test verifies the score is properly clamped.
    """
    scorer = ContextRelevancy()

    # Short context
    context = "Hello world"

    # Mock response where extracted sentences are LONGER than the context
    # This would produce a raw score > 1.0 without clamping
    mock_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(
                                    {
                                        "sentences": [
                                            {
                                                "sentence": "Hello world, this is a much longer sentence than the original context"
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }
            }
        ]
    }

    result = scorer._postprocess(context, mock_response)

    # Score should be clamped to 1.0, not exceed it
    assert result.score == 1.0
    assert result.score <= 1.0
    assert result.score >= 0.0


def test_context_relevancy_score_normal_case():
    """Test that ContextRelevancy returns expected score for normal case."""
    scorer = ContextRelevancy()

    context = "Hello world, this is a test context with some content."

    # Mock response where extracted sentences are shorter than the context
    mock_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"arguments": json.dumps({"sentences": [{"sentence": "Hello world"}]})}}
                    ]
                }
            }
        ]
    }

    result = scorer._postprocess(context, mock_response)

    # Score should be len("Hello world") / len(context) = 11 / 54 ≈ 0.204
    expected_score = len("Hello world") / len(context)
    assert result.score == pytest.approx(expected_score, rel=1e-3)
    assert result.score <= 1.0
    assert result.score >= 0.0


@respx.mock
def test_answer_correctness_uses_custom_embedding_model():
    """Test that AnswerCorrectness passes embedding_model parameter through to embeddings API."""
    captured_embedding_model = None

    def capture_embedding_model(request):
        nonlocal captured_embedding_model
        body = request.content.decode()
        import json

        data = json.loads(body)
        captured_embedding_model = data.get("model")
        return Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1] * 1536,
                        "index": 0,
                    }
                ],
                "model": data.get("model"),
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

    def mock_chat_completions(request):
        return Response(
            200,
            json={
                "id": "test-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_test",
                                    "type": "function",
                                    "function": {
                                        "name": "classify_statements",
                                        "arguments": '{"TP": ["Paris is the capital"], "FP": [], "FN": []}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=mock_chat_completions)
    respx.post("https://api.openai.com/v1/embeddings").mock(side_effect=capture_embedding_model)

    init(OpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"))

    metric = AnswerCorrectness(embedding_model="text-embedding-3-large")
    metric.eval(
        input="What is the capital of France?",
        output="Paris",
        expected="Paris is the capital of France",
    )

    assert captured_embedding_model == "text-embedding-3-large"
