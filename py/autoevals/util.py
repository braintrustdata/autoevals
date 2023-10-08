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


def traced(f=None, **span_kwargs):
    try:
        from braintrust.logger import traced as _traced

        return _traced(f, **span_kwargs)
    except ImportError:
        if f is None:
            return lambda f: f
        else:
            return f
