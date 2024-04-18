import asyncio
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any

PROXY_URL = "https://braintrustproxy.com/v1"


@dataclass
class ClaudeAIWrapper:
    complete: Any
    RateLimitError: Exception


def prepare_claude(is_async=False, api_key=None, base_url=None):
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("BRAINTRUST_API_KEY")
    if base_url is None:
        base_url = os.environ.get("ANTHROPIC_BASE_URL", PROXY_URL)
    try:
        import anthropic
    except Exception as e:
        print(
            textwrap.dedent(
                f"""\
            Unable to import anthropic: {e}

            Please install it, e.g. with

              pip install 'anthropic'
            """
            ),
            file=sys.stderr,
        )
        raise

    if is_async:
        anthropic_obj = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
    else:
        anthropic_obj = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    wrapper = ClaudeAIWrapper(
        complete=anthropic_obj.beta.tools.messages.create,
        RateLimitError=anthropic.RateLimitError,
    )

    return wrapper


def post_process_response(resp):
    return resp.dict()


def run_cached_request(request_type="complete", api_key=None, base_url=None, **kwargs):
    wrapper = prepare_claude(is_async=False, api_key=api_key, base_url=base_url)
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
    wrapper = prepare_claude(is_async=True, api_key=api_key, base_url=base_url)
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
