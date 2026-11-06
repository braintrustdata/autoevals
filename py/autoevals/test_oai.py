import sys
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from braintrust.oai import NamedWrapper, wrap_openai

from autoevals import init  # type: ignore[import]
from autoevals.oai import (  # type: ignore[import]
    GATEWAY_URL,
    LLMClient,
    OpenAIV0Module,
    OpenAIV1Module,
    _named_wrapper,  # type: ignore[import]  # Accessing private members for testing
    _wrap_openai,  # type: ignore[import]  # Accessing private members for testing
    arun_cached_request,
    get_default_model,
    get_openai_wrappers,
    prepare_openai,
    run_cached_request,
)


def unwrap_named_wrapper(obj: NamedWrapper | OpenAIV1Module.OpenAI | OpenAIV0Module) -> Any:
    return getattr(obj, "_NamedWrapper__wrapped", obj)


@pytest.fixture(autouse=True)
def reset_env_and_client(monkeypatch: pytest.MonkeyPatch):
    """Reset environment variables and client before each test."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://test-url")
    monkeypatch.delenv("BRAINTRUST_AI_GATEWAY_URL", raising=False)
    monkeypatch.setattr("autoevals.oai._named_wrapper", None)
    monkeypatch.setattr("autoevals.oai._wrap_openai", None)
    monkeypatch.setattr("autoevals.oai._openai_module", None)

    init(None)

    yield


def test_prepare_openai_uses_unwrapped_global_client():
    openai_obj = openai.OpenAI(api_key="api-key", base_url="http://test")
    client = LLMClient(
        openai=openai_obj,
        complete=openai_obj.chat.completions.create,
        embed=openai_obj.embeddings.create,
        moderation=openai_obj.moderations.create,
        RateLimitError=openai.RateLimitError,
    )

    init(client)

    prepared_client = prepare_openai()

    assert prepared_client == client
    assert not prepared_client.is_wrapped
    assert prepared_client.openai == openai_obj
    assert prepared_client.complete is client.complete
    assert prepared_client.openai.api_key == "api-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "custom_methods", [("complete", "embed", "moderation"), ("complete",), ("embed",), ("moderation",)]
)
async def test_custom_v1_methods_are_used_and_other_methods_default(is_async, custom_methods, mocker):
    request_args = {
        "complete": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
        "embed": {"model": "text-embedding-3-small", "input": "hello", "encoding_format": "float"},
        "moderation": {"model": "omni-moderation-latest", "input": "hello"},
    }
    responses = {
        "complete": {
            "id": "chat-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        },
        "embed": {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [1.0]}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
        "moderation": {"id": "mod-test", "model": "omni-moderation-latest", "results": []},
    }
    paths = {"/v1/chat/completions": "complete", "/v1/embeddings": "embed", "/v1/moderations": "moderation"}
    sdk_calls = []
    custom_calls = {method: [] for method in custom_methods}
    sleep = mocker.patch("autoevals.oai.time.sleep")
    async_sleep = mocker.patch("autoevals.oai.asyncio.sleep", new_callable=AsyncMock)

    class CustomRateLimitError(Exception):
        pass

    def respond(request):
        method = paths[request.url.path]
        sdk_calls.append(method)
        return httpx.Response(200, json=responses[method])

    def make_callback(method):
        def callback(**kwargs):
            custom_calls[method].append(kwargs)
            if len(custom_calls[method]) == 1:
                raise CustomRateLimitError("try again")
            return responses[method]

        async def async_callback(**kwargs):
            return callback(**kwargs)

        return async_callback if is_async else callback

    transport = httpx.MockTransport(respond)
    http_client = httpx.AsyncClient(transport=transport) if is_async else httpx.Client(transport=transport)
    sdk_class = openai.AsyncOpenAI if is_async else openai.OpenAI
    sdk = sdk_class(api_key="test-key", base_url="http://test/v1", http_client=http_client)
    try:
        client = LLMClient(
            openai=sdk,
            is_async=is_async,
            RateLimitError=CustomRateLimitError,
            **{method: make_callback(method) for method in custom_methods},
        )
        init(client)
        for method, kwargs in request_args.items():
            if is_async:
                result = await arun_cached_request(request_type=method, **kwargs)
            else:
                result = run_cached_request(request_type=method, **kwargs)
            assert result["model"] == kwargs["model"]
        assert custom_calls == {method: [request_args[method], request_args[method]] for method in custom_methods}
        assert sdk_calls == [method for method in request_args if method not in custom_methods]
        assert (async_sleep.await_count if is_async else sleep.call_count) == len(custom_methods)
    finally:
        if is_async:
            await sdk.close()
        else:
            sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "custom_methods", [("complete", "embed", "moderation"), ("complete",), ("embed",), ("moderation",)]
)
async def test_custom_v0_methods_are_used_and_other_methods_default(mock_openai_v0, is_async, custom_methods, mocker):
    request_args = {
        "complete": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
        "embed": {"model": "text-embedding-3-small", "input": "hello"},
        "moderation": {"input": "hello"},
    }

    class CustomRateLimitError(Exception):
        pass

    mock_type = AsyncMock if is_async else mocker.Mock
    defaults = {}
    for method, resource in (("complete", "ChatCompletion"), ("embed", "Embedding"), ("moderation", "Moderation")):
        defaults[method] = mock_type(return_value={"method": f"default-{method}"})
        mocker.patch.object(getattr(mock_openai_v0, resource), "acreate" if is_async else "create", defaults[method])
    callbacks = {
        method: mock_type(side_effect=[CustomRateLimitError("try again"), {"method": f"custom-{method}"}])
        for method in custom_methods
    }
    sleep = mocker.patch("autoevals.oai.time.sleep")
    async_sleep = mocker.patch("autoevals.oai.asyncio.sleep", new_callable=AsyncMock)
    client = LLMClient(openai=mock_openai_v0, is_async=is_async, RateLimitError=CustomRateLimitError, **callbacks)
    for method, kwargs in request_args.items():
        if is_async:
            result = await arun_cached_request(client=client, request_type=method, **kwargs)
        else:
            result = run_cached_request(client=client, request_type=method, **kwargs)
        assert result == {"method": f"{'custom' if method in custom_methods else 'default'}-{method}"}
        if method in custom_methods:
            assert callbacks[method].call_count == 2
            defaults[method].assert_not_called()
        else:
            defaults[method].assert_called_once_with(**kwargs)
    assert (async_sleep.await_count if is_async else sleep.call_count) == len(custom_methods)


def test_init_creates_llmclient_if_needed():
    openai_obj = openai.OpenAI()
    init(openai_obj)

    prepared_client = prepare_openai()

    assert isinstance(prepared_client, LLMClient)
    assert unwrap_named_wrapper(prepared_client.openai) == openai_obj


def test_init_creates_async_llmclient_if_needed(mock_openai_v0: OpenAIV0Module):
    init(mock_openai_v0, is_async=True)

    prepared_client = prepare_openai()

    assert isinstance(prepared_client, LLMClient)
    assert prepared_client.complete.__name__ == "acreate"


def test_prepare_openai_defaults():
    prepared_client = prepare_openai()

    assert isinstance(prepared_client, LLMClient)
    openai_obj = unwrap_named_wrapper(prepared_client.openai)
    assert isinstance(openai_obj, openai.OpenAI)
    assert callable(prepared_client.complete)
    assert prepared_client.complete.__name__ == "complete_wrapper"
    assert openai_obj.api_key == "test-key"
    assert openai_obj.base_url == "http://test-url"


def test_prepare_openai_defaults_to_gateway(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("BRAINTRUST_AI_GATEWAY_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("BRAINTRUST_API_KEY", "braintrust-key")

    prepared_client = prepare_openai()

    openai_obj = unwrap_named_wrapper(prepared_client.openai)
    assert openai_obj.api_key == "braintrust-key"
    assert str(openai_obj.base_url).rstrip("/") == GATEWAY_URL


def test_prepare_openai_uses_configured_gateway_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("BRAINTRUST_AI_GATEWAY_URL", "  https://gateway.example.com  ")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("BRAINTRUST_API_KEY", "braintrust-key")

    prepared_client = prepare_openai()

    openai_obj = unwrap_named_wrapper(prepared_client.openai)
    assert openai_obj.api_key == "braintrust-key"
    assert str(openai_obj.base_url).rstrip("/") == "https://gateway.example.com"


def test_prepare_openai_with_plain_openai():
    client = openai.OpenAI(api_key="api-key", base_url="http://test")
    prepared_client = prepare_openai(client=client)

    assert isinstance(prepared_client.openai, openai.OpenAI)


def test_prepare_openai_async():
    prepared_client = prepare_openai(is_async=True)

    assert isinstance(prepared_client, LLMClient)
    assert isinstance(prepared_client.openai, openai.AsyncOpenAI)

    assert callable(prepared_client.complete)
    assert prepared_client.complete.__name__ == "complete_wrapper"
    assert prepared_client.is_async


def test_prepare_openai_wraps_once():
    openai_obj = cast(OpenAIV1Module.OpenAI, wrap_openai(openai.OpenAI(api_key="api-key", base_url="http://test")))

    client = LLMClient(openai_obj)

    init(client)

    prepared_client = prepare_openai()

    assert prepared_client is client
    assert prepared_client.openai is openai_obj


def test_prepare_openai_handles_missing_braintrust(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "braintrust.oai", None)

    prepared_client = prepare_openai()

    assert isinstance(prepared_client, LLMClient)
    assert not prepared_client.is_wrapped
    assert isinstance(prepared_client.openai, openai.OpenAI)


def test_get_openai_wrappers_caches_imports():
    original_wrapper = _named_wrapper
    original_wrap_fn = _wrap_openai

    # First call should set the cache
    wrapper1, wrap_fn1 = get_openai_wrappers()

    # Second call should use cache
    wrapper2, wrap_fn2 = get_openai_wrappers()

    # Verify we got same objects back
    assert wrapper2 is wrapper1
    assert wrap_fn2 is wrap_fn1

    # Verify they're different from the original None values
    assert wrapper2 is not original_wrapper
    assert wrap_fn2 is not original_wrap_fn


def test_prepare_openai_raises_on_missing_openai(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(ImportError):
        prepare_openai()


@pytest.fixture
def mock_openai_v0(monkeypatch: pytest.MonkeyPatch):
    """Mock the OpenAI v0 SDK for testing."""

    class MockOpenAIV0:
        __module__ = "openai"
        api_key = None
        api_base = None

        class ChatCompletion:
            __module__ = "openai"

            @staticmethod
            def create(*args: Any, **kwargs: Any):
                pass

            @staticmethod
            def acreate(*args: Any, **kwargs: Any):
                pass

        class Embedding:
            __module__ = "openai"

            @staticmethod
            def create(*args: Any, **kwargs: Any):
                pass

            @staticmethod
            def acreate(*args: Any, **kwargs: Any):
                pass

        class Moderation:
            __module__ = "openai"

            @staticmethod
            def create(*args: Any, **kwargs: Any):
                pass

            @staticmethod
            def acreate(*args: Any, **kwargs: Any):
                pass

        class error:
            __module__ = "openai"

            class RateLimitError(Exception):
                __module__ = "openai"
                pass

    mock_openai = MockOpenAIV0()
    monkeypatch.setitem(sys.modules, "openai", mock_openai)
    return cast(OpenAIV0Module, mock_openai)


def test_prepare_openai_v0_sdk(mock_openai_v0: OpenAIV0Module):
    prepared_client = prepare_openai()

    assert prepared_client.openai.api_key == "test-key"
    assert prepared_client.complete.__name__ == "create"


def test_prepare_openai_v0_async(mock_openai_v0: OpenAIV0Module):
    prepared_client = prepare_openai(is_async=True)

    assert prepared_client.openai.api_key == "test-key"

    assert prepared_client.complete.__name__ == "acreate"


def test_prepare_openai_v0_with_client(mock_openai_v0: OpenAIV0Module):
    client = LLMClient(openai=mock_openai_v0, is_async=True)

    prepared_client = prepare_openai(client=client)

    assert prepared_client.openai.api_key is mock_openai_v0.api_key  # must be set by the user
    assert prepared_client.complete.__name__ == "acreate"


def test_get_default_model_returns_gpt_4o_by_default():
    """Test that get_default_model returns gpt-5-mini when no default is configured."""
    # Reset init to clear any previous default model
    init(None)
    assert get_default_model() == "gpt-5-mini"


def test_init_sets_default_model():
    """Test that init sets the default model correctly."""
    init(None, default_model="claude-3-5-sonnet-20241022")
    assert get_default_model() == "claude-3-5-sonnet-20241022"

    # Reset
    init(None)


def test_init_can_reset_default_model():
    """Test that init can reset the default model to gpt-5-mini."""
    init(None, default_model="claude-3-5-sonnet-20241022")
    assert get_default_model() == "claude-3-5-sonnet-20241022"

    init(None, default_model=None)
    assert get_default_model() == "gpt-5-mini"


def test_init_can_set_both_client_and_default_model():
    """Test that init can set both client and default model together."""
    client = openai.OpenAI(api_key="api-key", base_url="http://test")
    init(client, default_model="gpt-4-turbo")

    prepared_client = prepare_openai()

    # Unwrap to check it's the same client
    assert unwrap_named_wrapper(prepared_client.openai) == client

    assert get_default_model() == "gpt-4-turbo"

    # Reset
    init(None)
