import asyncio
from typing import cast
from unittest.mock import Mock

import pytest
import respx
from pydantic import BaseModel

from autoevals import init
from autoevals.llm import *
from autoevals.llm import build_classification_tools


class TestModel(BaseModel):
    foo: str
    num: int


def test_render_messages():
    classifier = OpenAILLMClassifier(
        "test",
        messages=[
            {"role": "user", "content": "{{value}} and {{{value}}}"},
            {"role": "user", "content": "Dict double braces: {{data}}"},
            {"role": "user", "content": "Dict triple braces: {{{data}}}"},
            {"role": "user", "content": "Model double braces: {{model}}"},
            {"role": "user", "content": "Model triple braces: {{{model}}}"},
            {"role": "user", "content": ""},  # test empty content
        ],
        model="gpt-4",
        choice_scores={"A": 1},
        classification_tools=[],
    )

    test_dict = {"foo": "bar", "num": 42}
    test_model = TestModel(foo="bar", num=42)

    rendered = classifier._render_messages(value="<b>bold</b>", data=test_dict, model=test_model)

    # Test that HTML is never escaped, regardless of syntax.
    assert rendered[0]["content"] == "<b>bold</b> and <b>bold</b>"

    # Test dict rendering - both use str().
    assert rendered[1]["content"] == "Dict double braces: {'foo': 'bar', 'num': 42}"
    assert rendered[2]["content"] == "Dict triple braces: {'foo': 'bar', 'num': 42}"

    # Test model rendering - both use str().
    assert rendered[3]["content"] == "Model double braces: foo='bar' num=42"
    assert rendered[4]["content"] == "Model triple braces: foo='bar' num=42"

    # Test empty content.
    assert rendered[5]["content"] == ""


def test_openai():
    e = OpenAILLMClassifier(
        "title",
        messages=[
            {
                "role": "system",
                "content": """\
You are a technical project manager who helps software engineers generate better titles for their GitHub issues.
You will look at the issue description, and pick which of two titles better describes it.""",
            },
            {
                "role": "user",
                "content": """\
I'm going to provide you with the issue description, and two possible titles.

Issue Description: {{page_content}}

1: {{output}}
2: {{expected}}

Please discuss each title briefly (one line for pros, one for cons), and then answer the question by calling
the select_choice function with "1" or "2".""",
            },
        ],
        model="gpt-3.5-turbo",
        choice_scores={"1": 1, "2": 0},
        classification_tools=build_classification_tools(useCoT=True, choice_strings=["1", "2"]),
        max_tokens=500,
    )

    page_content = """
As suggested by Nicolo, we should standardize the error responses coming from GoTrue, postgres, and realtime (and any other/future APIs) so that it's better DX when writing a client,

We can make this change on the servers themselves, but since postgrest and gotrue are fully/partially external may be harder to change, it might be an option to transform the errors within the client libraries/supabase-js, could be messy?

Nicolo also dropped this as a reference: http://spec.openapis.org/oas/v3.0.3#openapi-specification"""

    gen_title = "Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX"
    original_title = "This title has nothing to do with the content"

    response = e(gen_title, original_title, page_content=page_content)
    print(response.as_json(indent=2))
    assert response.score == 1
    assert response.error is None


def test_llm_classifier():
    for use_cot in [True, False]:
        e = LLMClassifier(
            "title",
            """
You are a technical project manager who helps software engineers generate better titles for their GitHub issues.
You will look at the issue description, and pick which of two titles better describes it.

I'm going to provide you with the issue description, and two possible titles.

Issue Description: {{page_content}}

1: {{output}}
2: {{expected}}""",
            {"1": 1, "2": 0},
            use_cot=use_cot,
        )

        page_content = """
As suggested by Nicolo, we should standardize the error responses coming from GoTrue, postgres, and realtime (and any other/future APIs) so that it's better DX when writing a client,

We can make this change on the servers themselves, but since postgrest and gotrue are fully/partially external may be harder to change, it might be an option to transform the errors within the client libraries/supabase-js, could be messy?

Nicolo also dropped this as a reference: http://spec.openapis.org/oas/v3.0.3#openapi-specification"""

        gen_title = "Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX"
        original_title = "This title has nothing to do with the content"

        response = e(gen_title, original_title, page_content=page_content)
        print(response.as_json(indent=2))
        assert response.score == 1
        assert response.error is None

        response = e(original_title, gen_title, page_content=page_content)
        print(response.as_json(indent=2))
        assert response.score == 0
        assert response.error is None


def test_nested_async():
    async def nested_async():
        e = Battle()
        e(instructions="Add the following numbers: 1, 2, 3", output="600", expected="6")

    asyncio.run(nested_async())


@respx.mock
def test_factuality():
    # something is wrong with respx that it couldn't match the url from openai
    respx.route().respond(
        json={
            "id": "chatcmpl-AdiS4bHWjqSclA5rx7OkuZ6EA9QIp",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": None,
                        "refusal": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_JKoeGAX2zGPJAmF2muDgjpHp",
                                "function": {
                                    "arguments": '{"reasons":"1. The question asks to add the numbers 1, 2, and 3.\\n2. The expert answer provides the sum of these numbers as 6.\\n3. The submitted answer also provides the sum as 6.\\n4. Both the expert and submitted answers provide the same numerical result, which is 6.\\n5. Since both answers provide the same factual content, the submitted answer contains all the same details as the expert answer.\\n6. There is no additional information or discrepancy between the two answers.\\n7. Therefore, the submitted answer is neither a subset nor a superset; it is exactly the same as the expert answer in terms of factual content.","choice":"C"}',
                                    "name": "select_choice",
                                },
                                "type": "function",
                            }
                        ],
                    },
                }
            ],
            "created": 1734029028,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion",
            "system_fingerprint": "fp_cc5cf1c6e3",
            "usage": {
                "completion_tokens": 149,
                "prompt_tokens": 404,
                "total_tokens": 553,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            },
        }
    )

    llm = Factuality(base_url="https://api.openai.com/v1/")
    result = llm.eval(
        output="6",
        expected="6",
        input="Add the following numbers: 1, 2, 3",
    )

    assert result.score == 1


def test_factuality_client():
    client = Mock()
    client.RateLimitError = Exception

    completion = Mock()
    completion.to_dict.return_value = {
        "id": "chatcmpl-AdiS4bHWjqSclA5rx7OkuZ6EA9QIp",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": None,
                    "refusal": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_JKoeGAX2zGPJAmF2muDgjpHp",
                            "function": {
                                "arguments": '{"reasons":"1. The question asks to add the numbers 1, 2, and 3.\\n2. The expert answer provides the sum of these numbers as 6.\\n3. The submitted answer also provides the sum as 6.\\n4. Both the expert and submitted answers provide the same numerical result, which is 6.\\n5. Since both answers provide the same factual content, the submitted answer contains all the same details as the expert answer.\\n6. There is no additional information or discrepancy between the two answers.\\n7. Therefore, the submitted answer is neither a subset nor a superset; it is exactly the same as the expert answer in terms of factual content.","choice":"C"}',
                                "name": "select_choice",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
        "created": 1734029028,
        "model": "gpt-4o-2024-08-06",
        "object": "chat.completion",
        "system_fingerprint": "fp_cc5cf1c6e3",
        "usage": {
            "completion_tokens": 149,
            "prompt_tokens": 404,
            "total_tokens": 553,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
    }

    client.complete.return_value = completion

    llm = Factuality(client=cast(LLMClient, client))
    result = llm.eval(
        output="6",
        expected="6",
        input="Add the following numbers: 1, 2, 3",
    )

    assert client.complete.call_count == 1

    assert result.score == 1


@pytest.fixture(autouse=True)
def reset_client():
    yield
    init(client=None)


# make sure we deny any leaked calls to OpenAI
@respx.mock(base_url="https://api.openai.com/v1/")
def test_init_client():
    client = Mock()
    client.RateLimitError = Exception

    completion = Mock()
    completion.to_dict.return_value = {
        "id": "chatcmpl-AdiS4bHWjqSclA5rx7OkuZ6EA9QIp",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": None,
                    "refusal": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_JKoeGAX2zGPJAmF2muDgjpHp",
                            "function": {
                                "arguments": '{"reasons":"1. The question asks to add the numbers 1, 2, and 3.\\n2. The expert answer provides the sum of these numbers as 6.\\n3. The submitted answer also provides the sum as 6.\\n4. Both the expert and submitted answers provide the same numerical result, which is 6.\\n5. Since both answers provide the same factual content, the submitted answer contains all the same details as the expert answer.\\n6. There is no additional information or discrepancy between the two answers.\\n7. Therefore, the submitted answer is neither a subset nor a superset; it is exactly the same as the expert answer in terms of factual content.","choice":"C"}',
                                "name": "select_choice",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
        "created": 1734029028,
        "model": "gpt-4o-2024-08-06",
        "object": "chat.completion",
        "system_fingerprint": "fp_cc5cf1c6e3",
        "usage": {
            "completion_tokens": 149,
            "prompt_tokens": 404,
            "total_tokens": 553,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
    }

    client.complete.return_value = completion

    init(client=client)

    llm = Factuality(base_url="https://api.openai.com/v1/")
    result = llm.eval(
        output="6",
        expected="6",
        input="Add the following numbers: 1, 2, 3",
    )

    assert client.complete.call_count == 1

    assert result.score == 1


def test_battle():
    for use_cot in [True, False]:
        print("use_cot", use_cot)
        e = Battle(use_cot=use_cot)
        response = e(
            instructions="Add the following numbers: 1, 2, 3",
            output="600",
            expected="6",
        )

        print(response.as_json(indent=2))
        assert response.score == 0
        assert response.error is None

        response = e(
            instructions="Add the following numbers: 1, 2, 3",
            output="6",
            expected="600",
        )

        print(response.as_json(indent=2))
        assert response.score == 1
        assert response.error is None

        response = e(instructions="Add the following numbers: 1, 2, 3", output="6", expected="6")

        print(response.as_json(indent=2))
        assert response.score == 0
        assert response.error is None
