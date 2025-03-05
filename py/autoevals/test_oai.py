import sys
from typing import Any, cast

import openai
import pytest
from braintrust.oai import (
    ChatCompletionV0Wrapper,
    CompletionsV1Wrapper,
    NamedWrapper,
    OpenAIV0Wrapper,
    OpenAIV1Wrapper,
    wrap_openai,
)
from openai.resources.chat.completions import AsyncCompletions

from autoevals import init  # type: ignore[import]
from autoevals.oai import (  # type: ignore[import]
    LLMClient,
    OpenAI,
    OpenAIV0Module,
    _named_wrapper,  # type: ignore[import]  # Accessing private members for testing
    _wrap_openai,  # type: ignore[import]  # Accessing private members for testing
    get_openai_wrappers,
    prepare_openai,
)


def unwrap_named_wrapper(obj: NamedWrapper | OpenAI | OpenAIV0Module) -> Any:
    return getattr(obj, "_NamedWrapper__wrapped")


@pytest.fixture(autouse=True)
def reset_env_and_client(monkeypatch: pytest.MonkeyPatch):
    """Reset environment variables and client before each test."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://test-url")
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


def test_init_creates_llmclient_if_needed():
    openai_obj = openai.OpenAI()
    init(openai_obj)

    prepared_client = prepare_openai()

    assert isinstance(prepared_client, LLMClient)
    assert prepared_client.is_wrapped
    assert unwrap_named_wrapper(prepared_client.openai) == openai_obj


def test_init_creates_async_llmclient_if_needed(mock_openai_v0: OpenAIV0Module):
    init(mock_openai_v0, is_async=True)

    prepared_client = prepare_openai()

    assert isinstance(prepared_client, LLMClient)
    assert prepared_client.is_wrapped
    assert isinstance(prepared_client.openai, OpenAIV0Wrapper)
    assert prepared_client.complete.__name__ == "acreate"


def test_prepare_openai_defaults():
    prepared_client = prepare_openai()

    assert isinstance(prepared_client, LLMClient)
    assert prepared_client.is_wrapped
    openai_obj = unwrap_named_wrapper(prepared_client.openai)
    assert isinstance(openai_obj, openai.OpenAI)
    assert isinstance(getattr(prepared_client.complete, "__self__", None), CompletionsV1Wrapper)
    assert openai_obj.api_key == "test-key"
    assert openai_obj.base_url == "http://test-url"


def test_prepare_openai_async():
    prepared_client = prepare_openai(is_async=True)

    assert isinstance(prepared_client, LLMClient)
    assert prepared_client.is_wrapped
    assert isinstance(prepared_client.openai, OpenAIV1Wrapper)

    openai_obj = getattr(prepared_client.complete, "__self__", None)
    assert isinstance(openai_obj, NamedWrapper)
    assert isinstance(unwrap_named_wrapper(openai_obj), AsyncCompletions)


def test_prepare_openai_wraps_once():
    openai_obj = cast(OpenAI, wrap_openai(openai.OpenAI(api_key="api-key", base_url="http://test")))

    client = LLMClient(openai_obj)

    init(client)

    prepared_client = prepare_openai()

    assert prepared_client is client
    assert prepared_client.is_wrapped
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

    assert prepared_client.is_wrapped
    assert prepared_client.openai.api_key == "test-key"

    assert isinstance(getattr(prepared_client.complete, "__self__", None), ChatCompletionV0Wrapper)


def test_prepare_openai_v0_async(mock_openai_v0: OpenAIV0Module):
    prepared_client = prepare_openai(is_async=True)

    assert prepared_client.is_wrapped
    assert prepared_client.openai.api_key == "test-key"

    assert prepared_client.complete.__name__ == "acreate"


def test_prepare_openai_v0_with_client(mock_openai_v0: OpenAIV0Module):
    client = LLMClient(openai=mock_openai_v0, is_async=True)

    prepared_client = prepare_openai(client=client)

    assert prepared_client.is_wrapped
    assert prepared_client.openai.api_key is mock_openai_v0.api_key  # must be set by the user
    assert prepared_client.complete.__name__ == "acreate"
