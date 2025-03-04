import asyncio
import importlib
import os
import sys
import textwrap
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

PROXY_URL = "https://api.braintrust.dev/v1/proxy"

_NAMED_WRAPPER: Optional[Type[Any]] = None
_WRAP_OPENAI: Optional[Callable[[Any], Any]] = None
_OPENAI_MODULE: Optional[Any] = None


def get_openai_module() -> Any:
    global _OPENAI_MODULE

    if _OPENAI_MODULE is not None:
        return _OPENAI_MODULE

    import openai

    _OPENAI_MODULE = openai
    return openai


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

    openai: Any
    complete: Callable[..., Any] = None  # type: ignore # Set in __post_init__
    embed: Callable[..., Any] = None  # type: ignore # Set in __post_init__
    moderation: Callable[..., Any] = None  # type: ignore # Set in __post_init__
    RateLimitError: Type[Exception] = None  # type: ignore # Set in __post_init__
    is_async: bool = False
    _is_wrapped: bool = False

    def __post_init__(self):
        NamedWrapper, wrap_openai = get_openai_wrappers()

        has_customization = self.complete is not None or self.embed is not None or self.moderation is not None

        # avoid wrapping if we have custom methods (the user may intend not to wrap)
        if not has_customization and not isinstance(self.openai, NamedWrapper):
            self.openai = wrap_openai(self.openai)

        self._is_wrapped = isinstance(self.openai, NamedWrapper)

        openai_module = get_openai_module()

        if hasattr(openai_module, "OpenAI"):
            # v1
            self.complete = self.openai.chat.completions.create
            self.embed = self.openai.embeddings.create
            self.moderation = self.openai.moderations.create
            self.RateLimitError = openai_module.RateLimitError
        else:
            # v0
            self.complete = self.openai.ChatCompletion.acreate if self.is_async else self.openai.ChatCompletion.create
            self.embed = self.openai.Embedding.acreate if self.is_async else self.openai.Embedding.create
            self.moderation = self.openai.Moderation.acreate if self.is_async else self.openai.Moderation.create
            self.RateLimitError = openai_module.error.RateLimitError

    @property
    def is_wrapped(self) -> bool:
        return self._is_wrapped


_client_var = ContextVar[Optional[LLMClient]]("client")


def get_openai_wrappers():
    global _NAMED_WRAPPER, _WRAP_OPENAI

    if _NAMED_WRAPPER is not None and _WRAP_OPENAI is not None:
        return _NAMED_WRAPPER, _WRAP_OPENAI

    try:
        from braintrust.oai import NamedWrapper as BraintrustNamedWrapper
        from braintrust.oai import wrap_openai

        _NAMED_WRAPPER = BraintrustNamedWrapper
    except ImportError:

        class NamedWrapper:
            pass

        def wrap_openai(openai: Any) -> Any:
            return openai

        _NAMED_WRAPPER = NamedWrapper

    _WRAP_OPENAI = wrap_openai
    return _NAMED_WRAPPER, _WRAP_OPENAI


def init(*, client: Optional[LLMClient] = None):
    """Initialize Autoevals with an optional custom LLM client.

    This function sets up the global client context for Autoevals to use. If no client is provided,
    the default OpenAI client will be used.

    Args:
        client (Optional[LLMClient]): A custom LLM client instance that implements the LLMClient interface.
            If None, the default OpenAI client will be used.\
    """
    _client_var.set(client)


def prepare_openai(client: Optional[LLMClient] = None, is_async=False, api_key=None, base_url=None):
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
        import openai
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

    if hasattr(openai, "OpenAI"):
        # v1 API
        if is_async:
            openai_obj = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            openai_obj = openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        # v0 API
        if api_key:
            openai.api_key = api_key
        openai.api_base = base_url
        openai_obj = openai

    return LLMClient(openai=openai_obj, is_async=is_async)


def post_process_response(resp):
    # This normalizes against craziness in OpenAI v0 vs. v1
    if hasattr(resp, "to_dict"):
        # v0
        return resp.to_dict()
    else:
        # v1
        return resp.dict()


def set_span_purpose(kwargs):
    kwargs.setdefault("span_info", {}).setdefault("span_attributes", {})["purpose"] = "scorer"


def run_cached_request(
    *, client: Optional[LLMClient] = None, request_type="complete", api_key=None, base_url=None, **kwargs
):
    wrapper = prepare_openai(client=client, is_async=False, api_key=api_key, base_url=base_url)
    if wrapper.is_wrapped:
        set_span_purpose(kwargs)

    retries = 0
    sleep_time = 0.1
    while retries < 100:
        try:
            resp = post_process_response(getattr(wrapper, request_type)(**kwargs))
            break
        except wrapper.RateLimitError:
            sleep_time *= 1.5
            time.sleep(sleep_time)
            retries += 1

    return resp


async def arun_cached_request(
    *, client: Optional[LLMClient] = None, request_type="complete", api_key=None, base_url=None, **kwargs
):
    wrapper = prepare_openai(client=client, is_async=True, api_key=api_key, base_url=base_url)
    if wrapper.is_wrapped:
        set_span_purpose(kwargs)

    retries = 0
    sleep_time = 0.1
    while retries < 100:
        try:
            resp = post_process_response(await getattr(wrapper, request_type)(**kwargs))
            break
        except wrapper.RateLimitError:
            # Just assume it's a rate limit error
            sleep_time *= 1.5
            await asyncio.sleep(sleep_time)
            retries += 1

    return resp
