import asyncio
import os
import sys
import textwrap
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

PROXY_URL = "https://api.braintrust.dev/v1/proxy"


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

    Note:
        If using async OpenAI methods you must use the async methods in Autoevals.

    Example:
        ```python
        # Using with OpenAI v1
        import openai
        client = LLMClient(
            openai=openai,
            complete=openai.chat.completions.create,
            embed=openai.embeddings.create,
            moderation=openai.moderations.create,
            RateLimitError=openai.RateLimitError
        )

        # Extending for custom implementation
        @dataclass
        class CustomLLMClient(LLMClient):
            def complete(self, **kwargs):
                # make adjustments as needed
                return openai.chat.completions.create(**kwargs)
        ```

    Note:
        This class is typically instantiated via the `prepare_openai()` function, which handles
        the SDK version detection and proper function assignment automatically.
    """

    openai: Any
    complete: Any
    embed: Any
    moderation: Any
    RateLimitError: Exception


_client_var = ContextVar[Optional[LLMClient]]("client")


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
    """Prepares and configures an OpenAI client for use with AutoEval, if client is not provided.

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
        Tuple[LLMClient, bool]: A tuple containing:
            - The configured LLMClient instance, or the client you've provided
            - A boolean indicating whether the client was wrapped with Braintrust tracing

    Raises:
        ImportError: If the OpenAI package is not installed
    """
    client = client or _client_var.get(None)

    openai = getattr(client, "openai", None)
    if not openai:
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

    openai_obj = openai

    is_v1 = False

    if hasattr(openai, "OpenAI"):
        # This is the new v1 API
        is_v1 = True

    if client is None:
        # prepare the default openai sdk, if not provided
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BRAINTRUST_API_KEY")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL", PROXY_URL)

        if is_v1:
            if is_async:
                openai_obj = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
            else:
                openai_obj = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            if api_key:
                openai.api_key = api_key
            openai.api_base = base_url

    # optimistically wrap openai instance for tracing
    wrapped = False
    try:
        from braintrust.oai import NamedWrapper, wrap_openai

        if not isinstance(openai_obj, NamedWrapper):
            openai_obj = wrap_openai(openai_obj)

        wrapped = True
    except ImportError:
        pass

    if client is None:
        # prepare the default client if not provided
        complete_fn = None
        rate_limit_error = None

        Client = LLMClient

        if is_v1:
            client = Client(
                openai=openai,
                complete=openai_obj.chat.completions.create,
                embed=openai_obj.embeddings.create,
                moderation=openai_obj.moderations.create,
                RateLimitError=openai.RateLimitError,
            )
        else:
            rate_limit_error = openai.error.RateLimitError
            if is_async:
                complete_fn = openai_obj.ChatCompletion.acreate
                embedding_fn = openai_obj.Embedding.acreate
                moderation_fn = openai_obj.Moderation.acreate
            else:
                complete_fn = openai_obj.ChatCompletion.create
                embedding_fn = openai_obj.Embedding.create
                moderation_fn = openai_obj.Moderation.create
            client = Client(
                openai=openai,
                complete=complete_fn,
                embed=embedding_fn,
                moderation=moderation_fn,
                RateLimitError=rate_limit_error,
            )

    return client, wrapped


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
    wrapper, wrapped = prepare_openai(client=client, is_async=False, api_key=api_key, base_url=base_url)
    if wrapped:
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
    wrapper, wrapped = prepare_openai(client=client, is_async=True, api_key=api_key, base_url=base_url)
    if wrapped:
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
