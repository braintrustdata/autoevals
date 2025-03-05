import asyncio
import os
import sys
import textwrap
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Type, TypeVar, Union, cast, runtime_checkable

PROXY_URL = "https://api.braintrust.dev/v1/proxy"


class OpenAIV1Module(Protocol):
    class OpenAI(Protocol):
        class chat(Protocol):
            class completions(Protocol):
                create: Callable[..., Any]

        class embeddings(Protocol):
            class create(Protocol):
                create: Callable[..., Any]

        class moderations(Protocol):
            class create(Protocol):
                create: Callable[..., Any]

        api_key: str

    class AsyncOpenAI(OpenAI):
        ...

    class RateLimitError(Exception):
        ...


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

    api_key: Optional[str]
    api_base: Optional[str]
    base_url: Optional[str]

    class error(Protocol):
        class RateLimitError(Exception):
            ...


_openai_module: Optional[OpenAIV1Module | OpenAIV0Module] = None


def get_openai_module() -> Union[OpenAIV1Module, OpenAIV0Module]:
    global _openai_module

    if _openai_module is not None:
        return _openai_module

    import openai  # type: ignore

    _openai_module = cast(Union[OpenAIV1Module, OpenAIV0Module], openai)
    return _openai_module


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
    RateLimitError: Type[Exception] = None  # type: ignore # Set in __post_init__
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
            self.complete = self.openai.chat.completions.create
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

T = TypeVar("T")

_named_wrapper: Optional[Type[Any]] = None
_wrap_openai: Optional[Callable[[Any], Any]] = None


def get_openai_wrappers() -> Tuple[Type[Any], Callable[[Any], Any]]:
    global _named_wrapper, _wrap_openai

    if _named_wrapper is not None and _wrap_openai is not None:
        return _named_wrapper, _wrap_openai

    try:
        from braintrust.oai import NamedWrapper as BraintrustNamedWrapper  # type: ignore
        from braintrust.oai import wrap_openai  # type: ignore

        _named_wrapper = cast(Type[Any], BraintrustNamedWrapper)
    except ImportError:

        class NamedWrapper:
            pass

        def wrap_openai(openai: T) -> T:
            return openai

        _named_wrapper = NamedWrapper

    _wrap_openai = cast(Callable[[Any], Any], wrap_openai)
    return _named_wrapper, _wrap_openai


def init(client: Optional[Union[LLMClient, OpenAIV0Module, OpenAIV1Module.OpenAI]] = None, is_async: bool = False):
    """Initialize Autoevals with an optional custom LLM client.

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
    """
    if client is None:
        configured_client = None
    elif isinstance(client, LLMClient):
        configured_client = client
    else:
        configured_client = LLMClient(client, is_async=is_async)

    _client_var.set(configured_client)


def prepare_openai(
    client: Optional[LLMClient] = None,
    is_async: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
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
        return client

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

    # prepare the default openai sdk, if not provided
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BRAINTRUST_API_KEY")
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL", PROXY_URL)

    if hasattr(openai_module, "OpenAI"):
        assert isinstance(openai_module, OpenAIV1Module)

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


def post_process_response(resp: Any) -> Dict[str, Any]:
    # This normalizes against craziness in OpenAI v0 vs. v1
    if hasattr(resp, "to_dict"):
        # v0
        return resp.to_dict()
    else:
        # v1
        return resp.dict()


def set_span_purpose(kwargs: Dict[str, Any]) -> None:
    kwargs.setdefault("span_info", {}).setdefault("span_attributes", {})["purpose"] = "scorer"


def run_cached_request(
    *,
    client: Optional[LLMClient] = None,
    request_type: str = "complete",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
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
    client: Optional[LLMClient] = None,
    request_type: str = "complete",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
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
