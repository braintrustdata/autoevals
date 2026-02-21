import asyncio
import os
import sys
import textwrap
import time
import warnings
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional, Protocol, TypedDict, TypeVar, Union, cast, runtime_checkable

PROXY_URL = "https://api.braintrust.dev/v1/proxy"


class DefaultModelConfig(TypedDict, total=False):
    """Configuration for default models used by Autoevals.

    This is used when passing the object form of `default_model` to `init()`.

    Attributes:
        completion: Default model for LLM-as-a-judge evaluations.
        embedding: Default model for embedding-based evaluations.
    """

    completion: str
    embedding: str


@runtime_checkable
class ChatCompletions(Protocol):
    create: Callable[..., Any]


@runtime_checkable
class Chat(Protocol):
    @property
    def completions(self) -> ChatCompletions: ...


@runtime_checkable
class Embeddings(Protocol):
    create: Callable[..., Any]


@runtime_checkable
class Moderations(Protocol):
    create: Callable[..., Any]


@runtime_checkable
class Responses(Protocol):
    create: Callable[..., Any]


@runtime_checkable
class OpenAIV1Module(Protocol):
    class OpenAI(Protocol):
        # Core API resources
        @property
        def chat(self) -> Chat: ...

        @property
        def embeddings(self) -> Embeddings: ...

        @property
        def moderations(self) -> Moderations: ...

        @property
        def responses(self) -> Responses: ...

        # Configuration
        @property
        def api_key(self) -> str: ...

        @property
        def organization(self) -> str | None: ...

        @property
        def base_url(self) -> str | Any | None: ...

    class AsyncOpenAI(OpenAI): ...

    class RateLimitError(Exception): ...


# TODO: we're removing v0 support in the next release
@runtime_checkable
class OpenAIV0Module(Protocol):
    class ChatCompletion(Protocol):
        acreate: Callable[..., Any]
        create: Callable[..., Any]

    class Embedding(Protocol):
        acreate: Callable[..., Any]
        create: Callable[..., Any]

    class Moderation(Protocol):
        acreate: Callable[..., Any]
        create: Callable[..., Any]

    api_key: str | None
    api_base: str | None
    base_url: str | None

    class error(Protocol):
        class RateLimitError(Exception): ...


_openai_module: OpenAIV1Module | OpenAIV0Module | None = None


def get_openai_module() -> OpenAIV1Module | OpenAIV0Module:
    global _openai_module

    if _openai_module is not None:
        return _openai_module

    import openai  # type: ignore

    _openai_module = cast(Union[OpenAIV1Module, OpenAIV0Module], openai)
    return _openai_module


def is_gpt5_model(model: str) -> bool:
    """Check if a model name indicates a GPT-5 class model."""
    return model.startswith("gpt-5")


@dataclass
class LLMClient:
    """A client wrapper for LLM operations that supports both OpenAI SDK v0 and v1.

    This class provides a consistent interface for common LLM operations regardless of the
    underlying OpenAI SDK version. It's designed to be extensible for custom implementations.

    Attributes:
        openai: The OpenAI module or client instance (either v0 or v1 SDK).
        complete: Completion function that creates chat completions.
            - For v0: openai.ChatCompletion.create or acreate
            - For v1: openai.chat.completions.create
        embed: Embedding function that creates embeddings.
            - For v0: openai.Embedding.create or acreate
            - For v1: openai.embeddings.create
        moderation: Moderation function that creates content moderations.
            - For v0: openai.Moderations.create or acreate
            - For v1: openai.moderations.create
        RateLimitError: The rate limit exception class for the SDK version.
            - For v0: openai.error.RateLimitError
            - For v1: openai.RateLimitError
        is_async: Whether the client is async (only used for v0 autoconfiguration).

    Note:
        If using async OpenAI methods you must use the async methods in Autoevals.
        The client will automatically configure itself if methods are not provided.

    Example:
        ```python
        # Using with OpenAI v1
        import openai
        client = openai.OpenAI()  # Configure with your settings
        llm = LLMClient(openai=client)  # Methods will be auto-configured

        # Or with explicit method configuration
        llm = LLMClient(
            openai=client,
            complete=client.chat.completions.create,
            embed=client.embeddings.create,
            moderation=client.moderations.create,
            RateLimitError=openai.RateLimitError
        )

        # Extending for custom implementation
        @dataclass
        class CustomLLMClient(LLMClient):
            def complete(self, **kwargs):
                # make adjustments as needed
                return self.openai.chat.completions.create(**kwargs)
        ```
    """

    openai: OpenAIV0Module | OpenAIV1Module.OpenAI
    complete: Callable[..., Any] = None  # type: ignore # Set in __post_init__
    embed: Callable[..., Any] = None  # type: ignore # Set in __post_init__
    moderation: Callable[..., Any] = None  # type: ignore # Set in __post_init__
    RateLimitError: type[Exception] = None  # type: ignore # Set in __post_init__
    is_async: bool = False
    _is_wrapped: bool = False

    def __post_init__(self):
        NamedWrapper, wrap_openai = get_openai_wrappers()

        has_customization = self.complete is not None or self.embed is not None or self.moderation is not None  # type: ignore  # Pyright doesn't understand our design choice

        # avoid wrapping if we have custom methods (the user may intend not to wrap)
        if not has_customization and not isinstance(self.openai, NamedWrapper):
            self.openai = wrap_openai(self.openai)

        self._is_wrapped = isinstance(self.openai, NamedWrapper)

        openai_module = get_openai_module()

        if hasattr(openai_module, "OpenAI"):
            openai_module = cast(OpenAIV1Module, openai_module)
            self.openai = cast(OpenAIV1Module.OpenAI, self.openai)

            # v1
            chat_complete = self.openai.chat.completions.create
            responses_create = self.openai.responses.create

            def complete_wrapper(**kwargs: Any) -> Any:
                model = kwargs.get("model", "")
                if is_gpt5_model(model):
                    responses_params = {
                        "model": kwargs["model"],
                        "input": kwargs["messages"],
                    }

                    # Transform tools from Chat Completions format to Responses API format
                    # Chat Completions: { type: "function", function: { name, description, parameters } }
                    # Responses API: { type: "function", name, description, parameters } (flattened)
                    if "tools" in kwargs:
                        tools = []
                        for tool in kwargs["tools"]:
                            if isinstance(tool, dict) and tool.get("type") == "function":
                                tools.append(
                                    {
                                        "type": "function",
                                        "name": tool["function"]["name"],
                                        "description": tool["function"].get("description"),
                                        "parameters": tool["function"].get("parameters"),
                                    }
                                )
                            else:
                                tools.append(tool)
                        responses_params["tools"] = tools

                    # Transform tool_choice format
                    # Chat Completions API: { type: "function", function: { name: "..." } }
                    # Responses API only accepts: "none", "auto", or "required"
                    if "tool_choice" in kwargs:
                        tool_choice = kwargs["tool_choice"]
                        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                            # Force the model to call a tool (equivalent to specifying a specific function)
                            responses_params["tool_choice"] = "required"
                        elif tool_choice in ["auto", "none"]:
                            responses_params["tool_choice"] = tool_choice
                        else:
                            # Default to required for other cases
                            responses_params["tool_choice"] = "required"

                    for key in ["temperature", "max_tokens", "reasoning_effort", "span_info"]:
                        if key in kwargs:
                            responses_params[key] = kwargs[key]
                    return responses_create(**responses_params)
                return chat_complete(**kwargs)

            self.complete = complete_wrapper
            self.embed = self.openai.embeddings.create
            self.moderation = self.openai.moderations.create
            self.RateLimitError = openai_module.RateLimitError
        else:
            openai_module = cast(OpenAIV0Module, openai_module)
            self.openai = cast(OpenAIV0Module, self.openai)

            # v0
            self.complete = self.openai.ChatCompletion.acreate if self.is_async else self.openai.ChatCompletion.create
            self.embed = self.openai.Embedding.acreate if self.is_async else self.openai.Embedding.create
            self.moderation = self.openai.Moderation.acreate if self.is_async else self.openai.Moderation.create
            self.RateLimitError = openai_module.error.RateLimitError

    @property
    def is_wrapped(self) -> bool:
        return self._is_wrapped


_client_var = ContextVar[Optional[LLMClient]]("client")
_default_model_var = ContextVar[Optional[str]]("default_model")
_default_embedding_model_var = ContextVar[Optional[str]]("default_embedding_model")

T = TypeVar("T")

_named_wrapper: type[Any] | None = None
_wrap_openai: Callable[[Any], Any] | None = None


def get_openai_wrappers() -> tuple[type[Any], Callable[[Any], Any]]:
    global _named_wrapper, _wrap_openai

    if _named_wrapper is not None and _wrap_openai is not None:
        return _named_wrapper, _wrap_openai

    try:
        from braintrust.oai import NamedWrapper as BraintrustNamedWrapper  # type: ignore
        from braintrust.oai import wrap_openai  # type: ignore

        _named_wrapper = cast(type[Any], BraintrustNamedWrapper)
    except ImportError:

        class NamedWrapper:
            pass

        def wrap_openai(openai: T) -> T:
            return openai

        _named_wrapper = NamedWrapper

    _wrap_openai = cast(Callable[[Any], Any], wrap_openai)
    return _named_wrapper, _wrap_openai


Client = Union[LLMClient, OpenAIV0Module, OpenAIV1Module.OpenAI]


def resolve_client(client: Client, is_async: bool = False) -> LLMClient:
    if isinstance(client, LLMClient):
        return client
    return LLMClient(openai=client, is_async=is_async)


def init(
    client: Client | None = None,
    is_async: bool = False,
    default_model: str | DefaultModelConfig | None = None,
):
    """Initialize Autoevals with an optional custom LLM client and default models.

    This function sets up the global client context for Autoevals to use. If no client is provided,
    the default OpenAI client will be used.

    Args:
        client: The client to use for LLM operations. Can be one of:
            - None: Resets the global client
            - LLMClient: Used directly as provided
            - OpenAIV0Module: Wrapped in a new LLMClient instance (OpenAI SDK v0)
            - OpenAIV1: Wrapped in a new LLMClient instance (OpenAI SDK v1)
        is_async: Whether to create a client with async operations. Defaults to False.
            Deprecated: Use the `client` argument directly with your desired async/sync configuration.
        default_model: The default model(s) to use for evaluations when not specified per-call.
            Can be either:
            - A string (for backward compatibility): Sets the default completion model only.
              Defaults to "gpt-5-mini" if not set.
            - A dictionary with "completion" and/or "embedding" keys: Allows setting default
              models for different evaluation types. Only the specified models are updated;
              others remain unchanged.

            When using non-OpenAI providers via the Braintrust proxy, set this to the
            appropriate model string (e.g., "claude-3-5-sonnet-20241022").

    Example:
        String form (backward compatible)::

            from autoevals import init
            init(default_model="gpt-4-turbo")

        Object form - set both models::

            from openai import OpenAI
            from autoevals import init

            init(
                client=OpenAI(
                    api_key=os.environ["BRAINTRUST_API_KEY"],
                    base_url="https://api.braintrust.dev/v1/proxy",
                ),
                default_model={
                    "completion": "claude-3-5-sonnet-20241022",
                    "embedding": "text-embedding-3-large",
                },
            )

        Object form - set only embedding model::

            init(
                default_model={
                    "embedding": "text-embedding-3-large",
                }
            )
    """
    _client_var.set(resolve_client(client, is_async=is_async) if client else None)

    if isinstance(default_model, str):
        # String form: sets completion model only, resets embedding to default
        _default_model_var.set(default_model)
        _default_embedding_model_var.set(None)
    elif default_model:
        # Object form: only update models that are explicitly provided
        if "completion" in default_model:
            _default_model_var.set(default_model["completion"])
        if "embedding" in default_model:
            _default_embedding_model_var.set(default_model["embedding"])
    else:
        # No default_model: reset both to defaults
        _default_model_var.set(None)
        _default_embedding_model_var.set(None)


def get_default_model() -> str:
    """Get the configured default completion model, or "gpt-5-mini" if not set."""
    return _default_model_var.get(None) or "gpt-5-mini"


def get_default_embedding_model() -> str:
    """Get the configured default embedding model, or "text-embedding-ada-002" if not set."""
    return _default_embedding_model_var.get(None) or "text-embedding-ada-002"


warned_deprecated_api_key_base_url = False


def prepare_openai(
    client: Client | None = None,
    is_async: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Prepares and configures an OpenAI client for use with AutoEval.

    This function handles both v0 and v1 of the OpenAI SDK, configuring the client
    with the appropriate authentication and base URL settings.

    We will also attempt to enable Braintrust tracing export, if you've configured tracing.

    Args:
        client (Optional[LLMClient], optional): Existing LLMClient instance.
            If provided, this client will be used instead of creating a new one.

        is_async (bool, optional): Whether to create a client with async operations. Defaults to False.
            Deprecated: Use the `client` argument and set the `openai` with the async/sync that you'd like to use.

        api_key (str, optional): OpenAI API key. If not provided, will look for
            OPENAI_API_KEY or BRAINTRUST_API_KEY in environment variables.
            Deprecated: Use the `client` argument and set the `openai`.

        base_url (str, optional): Base URL for API requests. If not provided, will
            use OPENAI_BASE_URL from environment or fall back to PROXY_URL.
            Deprecated: Use the `client` argument and set the `openai`.

    Returns:
        LLMClient: The configured LLMClient instance, or the client you've provided

    Raises:
        ImportError: If the OpenAI package is not installed
    """
    client = client or _client_var.get(None)
    if client is not None:
        return resolve_client(client, is_async=is_async)

    try:
        openai_module = get_openai_module()
    except Exception as e:
        print(
            textwrap.dedent(
                f"""\
            Unable to import openai: {e}

            Please install it, e.g. with

            pip install 'openai'
            """
            ),
            file=sys.stderr,
        )
        raise

    global warned_deprecated_api_key_base_url
    if not warned_deprecated_api_key_base_url and (api_key is not None or base_url is not None):
        warnings.warn(
            "The api_key and base_url parameters are deprecated. Please use init() or call with client instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        warned_deprecated_api_key_base_url = True

    # prepare the default openai sdk, if not provided
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BRAINTRUST_API_KEY")
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL", PROXY_URL)

    if hasattr(openai_module, "OpenAI"):
        openai_module = cast(OpenAIV1Module, openai_module)

        # v1 API
        if is_async:
            openai_obj = openai_module.AsyncOpenAI(api_key=api_key, base_url=base_url)  # type: ignore
        else:
            openai_obj = openai_module.OpenAI(api_key=api_key, base_url=base_url)  # type: ignore
    else:
        openai_module = cast(OpenAIV0Module, openai_module)

        # v0 API
        if api_key:
            openai_module.api_key = api_key
        openai_module.api_base = base_url
        openai_obj = openai_module

    return LLMClient(openai=openai_obj, is_async=is_async)


def post_process_response(resp: Any) -> dict[str, Any]:
    # This normalizes against craziness in OpenAI v0 vs. v1
    if hasattr(resp, "to_dict"):
        # v0
        return resp.to_dict()
    else:
        # v1
        return resp.dict()


def set_span_purpose(kwargs: dict[str, Any]) -> None:
    kwargs.setdefault("span_info", {}).setdefault("span_attributes", {})["purpose"] = "scorer"


def run_cached_request(
    *,
    client: LLMClient | None = None,
    request_type: str = "complete",
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    wrapper = prepare_openai(client=client, is_async=False, api_key=api_key, base_url=base_url)
    if wrapper.is_wrapped:
        set_span_purpose(kwargs)

    retries = 0
    sleep_time = 0.1
    resp = None
    while retries < 100:
        try:
            resp = post_process_response(getattr(wrapper, request_type)(**kwargs))
            break
        except wrapper.RateLimitError:
            sleep_time *= 1.5
            time.sleep(sleep_time)
            retries += 1

    if resp is None:
        raise RuntimeError("Failed to get response after maximum retries")
    return resp


async def arun_cached_request(
    *,
    client: LLMClient | None = None,
    request_type: str = "complete",
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    wrapper = prepare_openai(client=client, is_async=True, api_key=api_key, base_url=base_url)
    if wrapper.is_wrapped:
        set_span_purpose(kwargs)

    retries = 0
    sleep_time = 0.1
    resp = None
    while retries < 100:
        try:
            resp = post_process_response(await getattr(wrapper, request_type)(**kwargs))
            break
        except wrapper.RateLimitError:
            # Just assume it's a rate limit error
            sleep_time *= 1.5
            await asyncio.sleep(sleep_time)
            retries += 1

    if resp is None:
        raise RuntimeError("Failed to get response after maximum retries")

    return resp
