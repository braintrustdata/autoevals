import asyncio
import json
import os
import sqlite3
import sys
import textwrap
import threading
import time
from pathlib import Path

from .util import current_span

_CACHE_DIR = None
_CONN = None


def set_cache_dir(path):
    global _CACHE_DIR
    _CACHE_DIR = path


def open_cache():
    global _CACHE_DIR, _CONN
    if _CACHE_DIR is None:
        _CACHE_DIR = Path.home() / ".cache" / "braintrust"

    if _CONN is None:
        oai_cache_path = Path(_CACHE_DIR) / "oai.sqlite"
        os.makedirs(_CACHE_DIR, exist_ok=True)
        _CONN = sqlite3.connect(oai_cache_path, check_same_thread=False)
        _CONN.execute("CREATE TABLE IF NOT EXISTS cache (params text, response text)")
    return _CONN


CACHE_LOCK = threading.Lock()
PROXY_URL = "https://braintrustproxy.com/v1"


def prepare_openai_complete(is_async=False, api_key=None, base_url=None):
    if base_url is None:
        base_url = PROXY_URL

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
            openai_obj = openai.AsyncOpenAI(api_key=api_key, base_url=PROXY_URL)
        else:
            openai_obj = openai.OpenAI(api_key=api_key, base_url=PROXY_URL)
    else:
        if api_key:
            openai.api_key = api_key
        openai.api_base = PROXY_URL

    try:
        from braintrust.oai import wrap_openai

        openai_obj = wrap_openai(openai_obj)
    except ImportError:
        pass

    complete_fn = None
    rate_limit_error = None
    if is_v1:
        rate_limit_error = openai.RateLimitError
        complete_fn = openai_obj.chat.completions.create
    else:
        rate_limit_error = openai.error.RateLimitError
        if is_async:
            complete_fn = openai_obj.ChatCompletion.acreate
        else:
            complete_fn = openai_obj.ChatCompletion.create

    return complete_fn, rate_limit_error


def post_process_response(resp):
    # This normalizes against craziness in OpenAI v0 vs. v1
    if hasattr(resp, "to_dict"):
        # v0
        return resp.to_dict()
    else:
        # v1
        return resp.dict()


def log_cached_response(params, resp):
    with current_span().start_span(name="OpenAI Completion") as span:
        messages = params.pop("messages", None)
        span.log(
            metrics={
                "tokens": resp["usage"]["total_tokens"],
                "prompt_tokens": resp["usage"]["prompt_tokens"],
                "completion_tokens": resp["usage"]["completion_tokens"],
            },
            input=messages,
            output=resp["choices"],
        )


def run_cached_request(api_key=None, base_url=None, **kwargs):
    # OpenAI is very slow to import, so we only do it if we need it
    complete, RateLimitError = prepare_openai_complete(is_async=False, api_key=api_key, base_url=base_url)
    print(kwargs)

    retries = 0
    sleep_time = 0.1
    while retries < 100:
        try:
            resp = post_process_response(complete(**kwargs))
            break
        except RateLimitError:
            sleep_time *= 1.5
            time.sleep(sleep_time)
            retries += 1

    return resp


async def arun_cached_request(api_key=None, base_url=None, **kwargs):
    complete, RateLimitError = prepare_openai_complete(is_async=True, api_key=api_key, base_url=base_url)
    print(kwargs)

    retries = 0
    sleep_time = 0.1
    while retries < 100:
        try:
            resp = post_process_response(await complete(**kwargs))
            break
        except RateLimitError:
            # Just assume it's a rate limit error
            sleep_time *= 1.5
            await asyncio.sleep(sleep_time)
            retries += 1

    return resp
