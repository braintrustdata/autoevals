import dataclasses
import json
import sys
import textwrap


class SerializableDataClass:
    def as_dict(self):
        """Serialize the object to a dictionary."""
        return dataclasses.asdict(self)

    def as_json(self, **kwargs):
        """Serialize the object to JSON."""
        return json.dumps(self.as_dict(), **kwargs)


class NoOpSpan:
    def log(self, **kwargs):
        pass

    def start_span(self, *args, **kwargs):
        return self

    def end(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def current_span():
    try:
        from braintrust.logger import current_span as _get_current_span

        return _get_current_span()
    except ImportError as e:
        return NoOpSpan()


def traced(*span_args, **span_kwargs):
    try:
        from braintrust.logger import traced as _traced

        return _traced(*span_args, **span_kwargs)
    except ImportError:
        if len(span_args) == 1 and len(span_kwargs) == 0 and callable(span_args[0]):
            return span_args[0]
        else:
            return lambda f: f


def prepare_openai_complete(is_async=False, api_key=None):
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
    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
        # This is the new v1 API
        is_v1 = True
        if is_async:
            openai_obj = openai.AsyncOpenAI(api_key=api_key)
        else:
            openai_obj = openai.OpenAI(api_key=api_key)

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
