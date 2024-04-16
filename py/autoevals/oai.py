import asyncio
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROXY_URL = "https://braintrustproxy.com/v1"


@dataclass
class OpenAIWrapper:
    complete: Any
    embed: Any
    moderation: Any
    RateLimitError: Exception


def prepare_openai(is_async=False, api_key=None, base_url=None):
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BRAINTRUST_API_KEY")
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL", PROXY_URL)

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
        if is_async:
            openai_obj = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            openai_obj = openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        if api_key:
            openai.api_key = api_key
        openai.api_base = base_url

    try:
        from braintrust.oai import wrap_openai

        openai_obj = wrap_openai(openai_obj)
    except ImportError:
        pass

    complete_fn = None
    rate_limit_error = None
    if is_v1:
        wrapper = OpenAIWrapper(
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
            moderation_fn = openai_obj.Moderations.acreate
        else:
            complete_fn = openai_obj.ChatCompletion.create
            embedding_fn = openai_obj.Embedding.create
            moderation_fn = openai_obj.Moderations.create
        wrapper = OpenAIWrapper(
            complete=complete_fn,
            embed=embedding_fn,
            moderation=moderation_fn,
            RateLimitError=rate_limit_error,
        )

    return wrapper


def post_process_response(resp):
    # This normalizes against craziness in OpenAI v0 vs. v1
    if hasattr(resp, "to_dict"):
        # v0
        return resp.to_dict()
    else:
        # v1
        return resp.dict()


def run_cached_request(request_type="complete", api_key=None, base_url=None, **kwargs):
    wrapper = prepare_openai(is_async=False, api_key=api_key, base_url=base_url)

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


async def arun_cached_request(request_type="complete", api_key=None, base_url=None, **kwargs):
    wrapper = prepare_openai(is_async=True, api_key=api_key, base_url=base_url)

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
