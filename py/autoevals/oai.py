import asyncio
import sys
import textwrap
import time
from pathlib import Path

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


def run_cached_request(api_key=None, base_url=None, **kwargs):
    # OpenAI is very slow to import, so we only do it if we need it
    complete, RateLimitError = prepare_openai_complete(is_async=False, api_key=api_key, base_url=base_url)

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
