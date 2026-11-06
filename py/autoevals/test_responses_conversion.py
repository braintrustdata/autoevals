import pytest
import respx
from openai import AsyncOpenAI, OpenAI

from autoevals.oai import LLMClient, arun_cached_request, run_cached_request


def text_message(*texts):
    return {
        "id": "msg-test",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []} for text in texts],
    }


TOOL_CALLS = [
    {
        "type": "function_call",
        "id": "fc-test",
        "call_id": "call-0",
        "name": "select_choice",
        "arguments": '{"choice":"A"}',
    },
]
REQUEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "select_choice",
            "parameters": {"type": "object", "properties": {"choice": {"type": "string"}}},
        },
    },
]
CHAT_TOOL_IDENTITIES = [("call-0", "function", "select_choice")]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "output,content,tool_calls",
    [
        pytest.param(
            [text_message("first ", "second"), text_message(" third")],
            "first second third",
            [],
            id="message-text",
        ),
        pytest.param(
            [text_message("text with tools"), *TOOL_CALLS], "text with tools", CHAT_TOOL_IDENTITIES, id="mixed"
        ),
        pytest.param(TOOL_CALLS, None, CHAT_TOOL_IDENTITIES, id="tools-only"),
        pytest.param([{"type": "output_text", "text": "legacy text"}], "legacy text", [], id="top-level-output-text"),
        pytest.param([{"type": "text", "content": "legacy text"}], "legacy text", [], id="top-level-text"),
    ],
)
@respx.mock
async def test_responses_conversion(is_async, output, content, tool_calls):
    route = respx.post("https://api.openai.com/v1/responses").respond(
        json={
            "id": "resp-test",
            "object": "response",
            "created_at": 1234567890,
            "model": "gpt-5-mini",
            "status": "completed",
            "output": output,
        }
    )
    client_class = AsyncOpenAI if is_async else OpenAI
    client = client_class(api_key="test-api-key", max_retries=0)
    llm_client = LLMClient(openai=client, is_async=is_async)
    params = {"client": llm_client, "model": "gpt-5-mini", "messages": [{"role": "user", "content": "test"}]}
    if tool_calls:
        params["tools"] = REQUEST_TOOLS
    try:
        response = await arun_cached_request(**params) if is_async else run_cached_request(**params)
    finally:
        if is_async:
            await client.close()
        else:
            client.close()

    assert route.call_count == 1
    message = response["choices"][0]["message"]
    assert message["content"] == content
    converted_tools = message.get("tool_calls", [])
    assert [(tool["id"], tool["type"], tool["function"]["name"]) for tool in converted_tools] == tool_calls
    if tool_calls:
        assert converted_tools[0]["function"]["arguments"] == '{"choice":"A"}'
