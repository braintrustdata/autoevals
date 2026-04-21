"""Tests for the LiteLLM adapter clients."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from autoevals import init
from autoevals.litellm import AsyncLiteLLMClient, LiteLLMClient
from autoevals.oai import LLMClient


@pytest.fixture(autouse=True)
def reset_autoevals_state():
    init()
    yield
    init()


def _fake_completion_response(content: str = "hi") -> SimpleNamespace:
    """Minimal OpenAI-compatible completion response."""
    message = SimpleNamespace(content=content, role="assistant")
    choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
    return SimpleNamespace(choices=[choice], id="cmpl-1", model="test", object="chat.completion")


def test_litellm_client_exposes_openai_v1_surface():
    client = LiteLLMClient(api_key="sk-test", base_url="https://proxy.example/v1")
    # openai v1 protocol surface:
    assert hasattr(client.chat.completions, "create")
    assert hasattr(client.embeddings, "create")
    assert hasattr(client.moderations, "create")
    assert client.api_key == "sk-test"
    assert client.base_url == "https://proxy.example/v1"


def test_litellm_chat_completions_forwards_to_litellm(mocker):
    stub = mocker.patch("litellm.completion", return_value=_fake_completion_response("pong"))
    client = LiteLLMClient(api_key="sk-test", base_url="https://proxy.example/v1")

    resp = client.chat.completions.create(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "ping"}],
    )

    assert resp.choices[0].message.content == "pong"
    stub.assert_called_once()
    kwargs = stub.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude-3-5-sonnet-20241022"
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["api_base"] == "https://proxy.example/v1"


def test_litellm_client_without_api_key_does_not_forward_key(mocker):
    stub = mocker.patch("litellm.completion", return_value=_fake_completion_response())
    client = LiteLLMClient()  # LiteLLM will pick up env vars per provider

    client.chat.completions.create(model="openai/gpt-4o-mini", messages=[])
    kwargs = stub.call_args.kwargs
    assert "api_key" not in kwargs
    assert "api_base" not in kwargs


def test_litellm_embeddings_forwards_to_litellm(mocker):
    stub = mocker.patch("litellm.embedding", return_value={"data": [{"embedding": [0.1, 0.2]}]})
    client = LiteLLMClient(api_key="sk-test")

    client.embeddings.create(model="text-embedding-3-small", input="hello")

    stub.assert_called_once()
    assert stub.call_args.kwargs["model"] == "text-embedding-3-small"
    assert stub.call_args.kwargs["api_key"] == "sk-test"


def test_litellm_moderations_forwards_to_litellm(mocker):
    stub = mocker.patch("litellm.moderation", return_value={"results": [{"flagged": False}]})
    client = LiteLLMClient()

    client.moderations.create(input="some text")

    stub.assert_called_once()


@pytest.mark.asyncio
async def test_async_litellm_chat_completions_forwards(mocker):
    stub = mocker.patch("litellm.acompletion", new=AsyncMock(return_value=_fake_completion_response("async-pong")))
    client = AsyncLiteLLMClient(api_key="sk-test")

    resp = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "ping"}],
    )

    assert resp.choices[0].message.content == "async-pong"
    assert stub.await_count == 1


def test_init_accepts_litellm_client(mocker):
    """End-to-end: init(client=LiteLLMClient()) builds a usable LLMClient."""
    mocker.patch("litellm.completion", return_value=_fake_completion_response("init-ok"))

    init(client=LiteLLMClient(api_key="sk-test"))

    from autoevals.oai import prepare_openai

    wrapper = prepare_openai()
    assert isinstance(wrapper, LLMClient)
    # Calling through the wrapper should dispatch to litellm.completion
    result = wrapper.complete(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "ping"}])
    assert result.choices[0].message.content == "init-ok"
