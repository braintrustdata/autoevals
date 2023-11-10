import dataclasses
import json
import time


class SerializableDataClass:
    def as_dict(self):
        """Serialize the object to a dictionary."""
        return dataclasses.asdict(self)

    def as_json(self, **kwargs):
        """Serialize the object to JSON."""
        return json.dumps(self.as_dict(), **kwargs)


# DEVNOTE: This is copied from braintrust-sdk/py/src/braintrust/logger.py
class _NoopSpan:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def id(self):
        return ""

    @property
    def span_id(self):
        return ""

    @property
    def root_span_id(self):
        return ""

    def log(self, **event):
        pass

    def start_span(self, name, span_attributes={}, start_time=None, set_current=None, **event):
        return self

    def end(self, end_time=None):
        return end_time or time.time()

    def close(self, end_time=None):
        return self.end(end_time)

    def __enter__(self):
        return self

    def __exit__(self, type, value, callback):
        del type, value, callback


def current_span():
    try:
        from braintrust.logger import current_span as _get_current_span

        return _get_current_span()
    except ImportError as e:
        return _NoopSpan()


def traced(*span_args, **span_kwargs):
    try:
        from braintrust.logger import traced as _traced

        return _traced(*span_args, **span_kwargs)
    except ImportError:
        if len(span_args) == 1 and len(span_kwargs) == 0 and callable(span_args[0]):
            return span_args[0]
        else:
            return lambda f: f
