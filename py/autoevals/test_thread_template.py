import asyncio
import json

import pytest
import respx
from httpx import Response
from openai import AsyncOpenAI, OpenAI

from autoevals import LLMClassifier


@pytest.mark.parametrize(
    "template,thread,rendered,expected_calls,overrides",
    [
        ("{{#user_messages}}{{content}}{{/user_messages}}", [{"role": "user", "content": "Hello"}], "Hello", 1, {}),
        (
            "{{# user_messages }}{{content}}{{/ user_messages }}",
            [{"role": "user", "content": "Hello"}],
            "Hello",
            1,
            {},
        ),
        ("{{^user_messages}}No users{{/user_messages}}", [{"role": "user", "content": "Hello"}], "", 1, {}),
        ("{{^user_messages}}No users{{/user_messages}}", [], "No users", 1, {}),
        ("{{&first_message.content}}", [{"role": "user", "content": "<hello>"}], "<hello>", 1, {}),
        ("{{& first_message.content }}", [{"role": "user", "content": "Hello"}], "Hello", 1, {}),
        ("{{#thread_count}}Present{{/thread_count}}", [{"role": "user", "content": "Hello"}], "Present", 1, {}),
        ("{{first_message.content}}", [{"role": "user", "content": "Hello"}], "Hello", 1, {}),
        ("{{{first_message.content}}}", [{"role": "user", "content": "Hello"}], "Hello", 1, {}),
        ("Output: {{output}}", [], "Output: answer", 0, {}),
        ("{{!user_messages}}No conversation", [], "No conversation", 0, {}),
        (
            "{{#user_messages}}{{content}}{{/user_messages}}",
            [{"role": "user", "content": "Hello"}],
            "Override",
            1,
            {"user_messages": [{"role": "user", "content": "Override"}]},
        ),
        (
            "{{^user_messages}}No users{{/user_messages}}",
            [{"role": "user", "content": "Hello"}],
            "No users",
            1,
            {"user_messages": []},
        ),
        (
            "{{&first_message.content}}",
            [{"role": "user", "content": "Hello"}],
            "Override",
            1,
            {"first_message": {"content": "Override"}},
        ),
    ],
)
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
@respx.mock
async def test_thread_tags_fetch_and_render_trace(template, thread, rendered, expected_calls, overrides, use_async):
    class Trace:
        calls = 0

        async def get_thread(self):
            self.calls += 1
            return thread

    trace = Trace()
    route = respx.post("https://api.openai.com/v1/responses").mock(
        return_value=Response(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created": 1234567890,
                "model": "gpt-5-mini",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-test",
                        "name": "select_choice",
                        "arguments": '{"choice": "A"}',
                    }
                ],
            },
        )
    )
    client_type = AsyncOpenAI if use_async else OpenAI
    client = client_type(api_key="test-api-key", base_url="https://api.openai.com/v1")
    classifier = LLMClassifier(
        "thread-tags",
        "BEGIN" + template + "END",
        {"A": 1},
        model="gpt-5-mini",
        client=client,
        use_cot=False,
    )
    try:
        if use_async:
            score = await classifier.eval_async(output="answer", trace=trace, **overrides)
        else:
            score = await asyncio.to_thread(classifier.eval, output="answer", trace=trace, **overrides)
    finally:
        if use_async:
            await client.close()
        else:
            client.close()

    assert score.score == 1
    assert trace.calls == expected_calls
    content = json.loads(route.calls.last.request.content)["input"][0]["content"]
    assert content == (
        "BEGIN" + rendered + "END\n" "Answer the question by calling `select_choice` with a single choice from ['A']."
    )
