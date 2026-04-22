"""LiteLLM adapters — route Autoevals through the LiteLLM AI gateway for direct
access to 100+ LLM providers (OpenAI, Anthropic, Bedrock, Vertex, Gemini, Ollama,
OpenRouter, Groq, DeepSeek, etc.) using provider-native API keys.

Example::

    from autoevals import init
    from autoevals.litellm import LiteLLMClient
    from autoevals.llm import Factuality

    init(
        client=LiteLLMClient(),
        default_model="anthropic/claude-3-5-sonnet-20241022",
    )

    evaluator = Factuality()
    result = evaluator.eval(input="...", output="...", expected="...")

Unlike the Braintrust AI Proxy path (which requires a Braintrust API key), LiteLLM
uses each provider's native key (``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``,
``AWS_*``, etc.) and routes locally. See https://docs.litellm.ai/docs/providers.
"""

from __future__ import annotations

from typing import Any, Optional


class _LiteLLMChatCompletions:
    """Sync ``openai.chat.completions`` surface backed by ``litellm.completion``."""

    def __init__(self, api_key: Optional[str], base_url: Optional[str]):
        self._api_key = api_key
        self._base_url = base_url

    def create(self, **kwargs: Any) -> Any:
        import litellm

        if self._api_key is not None:
            kwargs.setdefault("api_key", self._api_key)
        if self._base_url is not None:
            kwargs.setdefault("api_base", self._base_url)
        return litellm.completion(**kwargs)


class _AsyncLiteLLMChatCompletions:
    """Async counterpart of ``_LiteLLMChatCompletions``."""

    def __init__(self, api_key: Optional[str], base_url: Optional[str]):
        self._api_key = api_key
        self._base_url = base_url

    async def create(self, **kwargs: Any) -> Any:
        import litellm

        if self._api_key is not None:
            kwargs.setdefault("api_key", self._api_key)
        if self._base_url is not None:
            kwargs.setdefault("api_base", self._base_url)
        return await litellm.acompletion(**kwargs)


class _LiteLLMChat:
    def __init__(self, completions: Any):
        self.completions = completions


class _LiteLLMEmbeddings:
    def __init__(self, api_key: Optional[str], base_url: Optional[str], is_async: bool):
        self._api_key = api_key
        self._base_url = base_url
        self._is_async = is_async

    def _kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self._api_key is not None:
            kwargs.setdefault("api_key", self._api_key)
        if self._base_url is not None:
            kwargs.setdefault("api_base", self._base_url)
        return kwargs

    def create(self, **kwargs: Any) -> Any:
        import litellm

        if self._is_async:
            return litellm.aembedding(**self._kwargs(kwargs))
        return litellm.embedding(**self._kwargs(kwargs))


class _LiteLLMModerations:
    def __init__(self, api_key: Optional[str], base_url: Optional[str], is_async: bool):
        self._api_key = api_key
        self._base_url = base_url
        self._is_async = is_async

    def _kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self._api_key is not None:
            kwargs.setdefault("api_key", self._api_key)
        if self._base_url is not None:
            kwargs.setdefault("api_base", self._base_url)
        return kwargs

    def create(self, **kwargs: Any) -> Any:
        import litellm

        if self._is_async:
            return litellm.amoderation(**self._kwargs(kwargs))
        return litellm.moderation(**self._kwargs(kwargs))


def _responses_params_to_chat_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate autoevals' Responses-API kwargs back into Chat-Completions kwargs
    that ``litellm.completion`` understands.

    autoevals' ``oai.py`` routes GPT-5 models through ``client.responses.create``
    (see ``is_gpt5_model``, ``prepare_responses_params``). Those params use
    ``input=`` instead of ``messages=`` and a Responses-API tool schema.
    ``litellm.completion`` only speaks Chat-Completions, so we translate back.
    The resulting ChatCompletion response is then detected by autoevals'
    ``convert_responses_to_chat_completion`` as not-a-Responses-object and
    returned as-is (see ``oai.py:226``).
    """
    chat_kwargs = dict(kwargs)
    if "input" in chat_kwargs and "messages" not in chat_kwargs:
        chat_kwargs["messages"] = chat_kwargs.pop("input")
    # Responses-API tools use flat {type, name, description, parameters}; Chat-
    # Completions tools nest the schema under {type, function: {...}}.
    if "tools" in chat_kwargs:
        translated = []
        for tool in chat_kwargs["tools"]:
            if isinstance(tool, dict) and tool.get("type") == "function" and "function" not in tool:
                translated.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description"),
                            "parameters": tool.get("parameters"),
                        },
                    }
                )
            else:
                translated.append(tool)
        chat_kwargs["tools"] = translated
    return chat_kwargs


class _LiteLLMResponses:
    """Adapter for autoevals' Responses-API code path (triggered by GPT-5 models).

    Without this, ``init(client=LiteLLMClient())`` with autoevals' default model
    (``gpt-5-mini``) would call ``litellm.completion(input=..., model=...)`` and
    crash because LiteLLM requires ``messages=``.
    """

    def __init__(self, api_key: Optional[str], base_url: Optional[str], is_async: bool):
        self._api_key = api_key
        self._base_url = base_url
        self._is_async = is_async

    def _kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        chat_kwargs = _responses_params_to_chat_kwargs(kwargs)
        if self._api_key is not None:
            chat_kwargs.setdefault("api_key", self._api_key)
        if self._base_url is not None:
            chat_kwargs.setdefault("api_base", self._base_url)
        return chat_kwargs

    def create(self, **kwargs: Any) -> Any:
        import litellm

        if self._is_async:
            return litellm.acompletion(**self._kwargs(kwargs))
        return litellm.completion(**self._kwargs(kwargs))


class _LiteLLMResponsesContainer:
    """Exposes ``.create`` on the ``client.responses`` attribute to match the
    OpenAI v1 client shape autoevals' ``oai.py`` duck-types on."""

    def __init__(self, create_impl: Any):
        self.create = create_impl


class LiteLLMClient:
    """OpenAI-compatible client backed by ``litellm.completion``.

    Pass to ``autoevals.init(client=LiteLLMClient())``. Routes every chat/embedding/
    moderation call through LiteLLM, which resolves the target provider from the
    model-name prefix (e.g. ``anthropic/claude-3-5-sonnet``, ``bedrock/anthropic.claude-3-sonnet``).

    Args:
        api_key: Optional provider API key. If unset, LiteLLM falls back to the
            per-provider env vars (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.).
        base_url: Optional custom base URL (forwarded to LiteLLM as ``api_base``).
        organization: Accepted for OpenAI-protocol compatibility; not used.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.chat = _LiteLLMChat(_LiteLLMChatCompletions(api_key=api_key, base_url=base_url))
        self.embeddings = _LiteLLMEmbeddings(api_key=api_key, base_url=base_url, is_async=False)
        self.moderations = _LiteLLMModerations(api_key=api_key, base_url=base_url, is_async=False)
        self.responses = _LiteLLMResponsesContainer(
            _LiteLLMResponses(api_key=api_key, base_url=base_url, is_async=False).create,
        )


class AsyncLiteLLMClient:
    """Async variant of :class:`LiteLLMClient` — uses ``litellm.acompletion`` etc."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.chat = _LiteLLMChat(_AsyncLiteLLMChatCompletions(api_key=api_key, base_url=base_url))
        self.embeddings = _LiteLLMEmbeddings(api_key=api_key, base_url=base_url, is_async=True)
        self.moderations = _LiteLLMModerations(api_key=api_key, base_url=base_url, is_async=True)
        self.responses = _LiteLLMResponsesContainer(
            _LiteLLMResponses(api_key=api_key, base_url=base_url, is_async=True).create,
        )
