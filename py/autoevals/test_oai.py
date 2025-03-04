import sys

import openai
import pytest
from braintrust.oai import (
    AsyncCompletionsV1Wrapper,
    ChatCompletionWrapper,
    CompletionsV1Wrapper,
    OpenAIV1Wrapper,
    wrap_openai,
)

from autoevals import init
from autoevals.oai import _NAMED_WRAPPER, _WRAP_OPENAI, LLMClient, get_openai_wrappers, prepare_openai


@pytest.fixture(autouse=True)
def reset_env_and_client(monkeypatch: pytest.MonkeyPatch):
    """Reset environment variables and client before each test."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://test-url")

    init(client=None)

    yield


def test_prepare_openai_uses_global_client():
    openai_obj = openai.OpenAI(api_key="api-key", base_url="http://test")
    client = LLMClient(
        openai=openai_obj,
        complete=openai_obj.chat.completions.create,
        embed=openai_obj.embeddings.create,
        moderation=openai_obj.moderations.create,
        RateLimitError=openai.RateLimitError,
    )

    init(client=client)

    prepared_client, wrapped = prepare_openai()

    assert prepared_client == client
    assert wrapped is False
    assert prepared_client.openai == openai_obj
    assert prepared_client.complete is client.complete
    assert prepared_client.openai.api_key == "api-key"
    assert prepared_client.openai.base_url == "http://test"


def test_prepare_openai_defaults(monkeypatch: pytest.MonkeyPatch):
    prepared_client, wrapped = prepare_openai()

    assert isinstance(prepared_client, LLMClient)
    assert wrapped is True
    openai_obj = getattr(prepared_client.openai, "_NamedWrapper__wrapped")
    assert isinstance(openai_obj, openai.OpenAI)
    assert isinstance(getattr(prepared_client.complete, "__self__", None), CompletionsV1Wrapper)
    assert openai_obj.api_key == "test-key"
    assert openai_obj.base_url == "http://test-url"


def test_prepare_openai_async():
    prepared_client, wrapped = prepare_openai(is_async=True)

    assert isinstance(prepared_client, LLMClient)
    assert wrapped is True
    assert isinstance(prepared_client.openai, OpenAIV1Wrapper)
    assert callable(prepared_client.complete)
    assert isinstance(getattr(prepared_client.complete, "__self__", None), AsyncCompletionsV1Wrapper)


def test_prepare_openai_wraps_once():
    openai_obj = wrap_openai(openai.OpenAI(api_key="api-key", base_url="http://test"))

    client = LLMClient(
        openai=openai_obj,
        complete=openai_obj.chat.completions.create,
        embed=openai_obj.embeddings.create,
        moderation=openai_obj.moderations.create,
        RateLimitError=openai.RateLimitError,
    )

    init(client=client)

    prepared_client, wrapped = prepare_openai()

    assert prepared_client is client
    assert wrapped is True
    assert prepared_client.openai is openai_obj


def test_prepare_openai_handles_missing_braintrust(monkeypatch):
    monkeypatch.setattr("autoevals.oai._NAMED_WRAPPER", None)
    monkeypatch.setattr("autoevals.oai._WRAP_OPENAI", None)
    monkeypatch.setitem(sys.modules, "braintrust.oai", None)

    prepared_client, wrapped = prepare_openai()

    assert isinstance(prepared_client, LLMClient)
    assert wrapped is False
    assert isinstance(prepared_client.openai, openai.OpenAI)


def test_get_openai_wrappers_caches_imports():
    original_wrapper = _NAMED_WRAPPER
    original_wrap_fn = _WRAP_OPENAI

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
