"""The RAGAS scorers must not fall back to their own hardcoded model.

`_get_model` is what every RAGAS scorer calls to resolve its judge model.
Before this change it returned `DEFAULT_RAGAS_MODEL` ("gpt-5-nano") when the
caller passed no model and had not called `init(default_model=...)`, so a
RAGAS scorer silently used a different judge than every other scorer in the
library -- and than the TypeScript implementation, which resolves through
`getDefaultModel()`. The comment above the constant already said the
hardcoded fallback was deprecated.
"""

from autoevals import init
from autoevals.oai import get_default_model
from autoevals.ragas import _get_model


def test_ragas_default_matches_library_default():
    init()
    assert _get_model(None) == get_default_model()


def test_ragas_respects_explicit_model():
    init()
    assert _get_model("gpt-4o") == "gpt-4o"


def test_ragas_respects_configured_default():
    init(default_model="gpt-4-turbo")
    assert _get_model(None) == "gpt-4-turbo"
