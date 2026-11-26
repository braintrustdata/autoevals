import json
import os

import pytest

from autoevals.thread_utils import compute_thread_template_vars, format_message_array_as_text

# Shared with js/thread-utils.test.ts so both languages render threads identically.
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "test-data", "thread_formatting.json")
with open(FIXTURE_PATH, encoding="utf-8") as f:
    FIXTURES = json.load(f)


@pytest.mark.parametrize("name", [n for n, c in FIXTURES.items() if "text" in c])
def test_format_message_array_matches_shared_fixture(name):
    case = FIXTURES[name]
    assert format_message_array_as_text(case["messages"]) == case["text"]


@pytest.mark.parametrize("name", [n for n, c in FIXTURES.items() if "message_text" in c])
def test_single_message_rendering_matches_shared_fixture(name):
    case = FIXTURES[name]
    first_message = compute_thread_template_vars(case["messages"])["first_message"]
    assert str(first_message) == case["message_text"]
