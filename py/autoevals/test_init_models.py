"""Tests for init() with default_model parameter supporting both string and object forms."""

import pytest

from autoevals import init
from autoevals.oai import get_default_embedding_model, get_default_model


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test."""
    init()
    yield
    init()


def test_string_form_sets_completion_model_backward_compatible():
    """Test that string form sets completion model (backward compatible)."""
    init(default_model="gpt-4-turbo")

    assert get_default_model() == "gpt-4-turbo"
    assert get_default_embedding_model() == "text-embedding-ada-002"  # Default


def test_object_form_can_set_completion_model_only():
    """Test that object form can set only completion model."""
    init(default_model={"completion": "gpt-4-turbo"})

    assert get_default_model() == "gpt-4-turbo"


def test_object_form_can_set_embedding_model_only():
    """Test that object form can set only embedding model."""
    init(default_model={"embedding": "text-embedding-3-large"})

    assert get_default_embedding_model() == "text-embedding-3-large"
    # Completion model should remain at default since we didn't update it
    assert get_default_model() == "gpt-4o"


def test_object_form_can_set_both_models():
    """Test that object form can set both models."""
    init(
        default_model={
            "completion": "claude-3-5-sonnet-20241022",
            "embedding": "text-embedding-3-large",
        }
    )

    assert get_default_model() == "claude-3-5-sonnet-20241022"
    assert get_default_embedding_model() == "text-embedding-3-large"


def test_partial_updates_preserve_unspecified_models():
    """Test that partial updates preserve models that are not explicitly set."""
    # First set completion model
    init(default_model={"completion": "gpt-4-turbo"})

    assert get_default_model() == "gpt-4-turbo"
    assert get_default_embedding_model() == "text-embedding-ada-002"

    # Then set only embedding model - completion should remain unchanged
    init(default_model={"embedding": "text-embedding-3-large"})

    assert get_default_model() == "gpt-4-turbo"  # Should still be gpt-4-turbo
    assert get_default_embedding_model() == "text-embedding-3-large"


def test_falls_back_to_defaults_when_not_set():
    """Test that defaults are used when default_model is not provided."""
    init()

    assert get_default_model() == "gpt-4o"
    assert get_default_embedding_model() == "text-embedding-ada-002"


def test_string_form_resets_embedding_model_to_default():
    """Test that string form resets embedding model to default."""
    # First set both models
    init(
        default_model={
            "completion": "gpt-4-turbo",
            "embedding": "text-embedding-3-large",
        }
    )

    assert get_default_model() == "gpt-4-turbo"
    assert get_default_embedding_model() == "text-embedding-3-large"

    # Then use string form - should reset embedding to default
    init(default_model="claude-3-5-sonnet-20241022")

    assert get_default_model() == "claude-3-5-sonnet-20241022"
    assert get_default_embedding_model() == "text-embedding-ada-002"  # Reset to default
