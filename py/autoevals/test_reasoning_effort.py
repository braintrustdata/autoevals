"""Tests for reasoning parameter support."""

import pytest

from autoevals.llm import LLMClassifier


def test_reasoning_effort_in_constructor():
    """Test that LLMClassifier accepts reasoning_effort parameter."""
    classifier = LLMClassifier(
        name="test",
        prompt_template="Evaluate: {{output}}",
        choice_scores={"good": 1, "bad": 0},
        model="o3-mini",
        reasoning_effort="high",
    )

    assert classifier is not None
    assert classifier.extra_args.get("reasoning_effort") == "high"


def test_reasoning_effort_values():
    """Test that all valid reasoning_effort values are accepted."""
    valid_values = ["minimal", "low", "medium", "high", None]

    for value in valid_values:
        classifier = LLMClassifier(
            name="test",
            prompt_template="Evaluate: {{output}}",
            choice_scores={"good": 1, "bad": 0},
            model="o3-mini",
            reasoning_effort=value,
        )

        assert classifier is not None
        if value is not None:
            assert classifier.extra_args.get("reasoning_effort") == value
        else:
            assert "reasoning_effort" not in classifier.extra_args


def test_reasoning_effort_not_set_by_default():
    """Test that reasoning_effort is not set when not provided."""
    classifier = LLMClassifier(
        name="test",
        prompt_template="Evaluate: {{output}}",
        choice_scores={"good": 1, "bad": 0},
        model="o3-mini",
    )

    assert "reasoning_effort" not in classifier.extra_args


def test_reasoning_enabled_in_constructor():
    """Test that LLMClassifier accepts reasoning_enabled parameter."""
    classifier = LLMClassifier(
        name="test",
        prompt_template="Evaluate: {{output}}",
        choice_scores={"good": 1, "bad": 0},
        model="claude-3-5-sonnet-20241022",
        reasoning_enabled=True,
    )

    assert classifier is not None
    assert classifier.extra_args.get("reasoning_enabled") is True


def test_reasoning_budget_in_constructor():
    """Test that LLMClassifier accepts reasoning_budget parameter."""
    classifier = LLMClassifier(
        name="test",
        prompt_template="Evaluate: {{output}}",
        choice_scores={"good": 1, "bad": 0},
        model="claude-3-5-sonnet-20241022",
        reasoning_budget=2048,
    )

    assert classifier is not None
    assert classifier.extra_args.get("reasoning_budget") == 2048


def test_all_reasoning_parameters():
    """Test that all reasoning parameters can be used together."""
    classifier = LLMClassifier(
        name="test",
        prompt_template="Evaluate: {{output}}",
        choice_scores={"good": 1, "bad": 0},
        model="o3-mini",
        reasoning_effort="high",
        reasoning_enabled=True,
        reasoning_budget=4096,
    )

    assert classifier is not None
    assert classifier.extra_args.get("reasoning_effort") == "high"
    assert classifier.extra_args.get("reasoning_enabled") is True
    assert classifier.extra_args.get("reasoning_budget") == 4096


def test_reasoning_enabled_not_set_by_default():
    """Test that reasoning_enabled is not set when not provided."""
    classifier = LLMClassifier(
        name="test",
        prompt_template="Evaluate: {{output}}",
        choice_scores={"good": 1, "bad": 0},
        model="claude-3-5-sonnet-20241022",
    )

    assert "reasoning_enabled" not in classifier.extra_args


def test_reasoning_budget_not_set_by_default():
    """Test that reasoning_budget is not set when not provided."""
    classifier = LLMClassifier(
        name="test",
        prompt_template="Evaluate: {{output}}",
        choice_scores={"good": 1, "bad": 0},
        model="claude-3-5-sonnet-20241022",
    )

    assert "reasoning_budget" not in classifier.extra_args
