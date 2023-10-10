import dataclasses
import json


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
