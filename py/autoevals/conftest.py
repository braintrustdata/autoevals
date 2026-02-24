"""Pytest configuration and fixtures for autoevals tests."""

import pytest
import respx
from httpx import Response


@pytest.fixture(autouse=True)
def mock_responses_api():
    """Automatically mock the OpenAI Responses API for all tests."""
    # Add a default mock for the Responses API endpoint
    # Individual tests can override this with more specific mocks
    route = respx.route(method="POST", path__regex=r".*/responses$")

    def responses_handler(request):
        """Default handler that returns a Responses API format response."""
        return Response(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created": 1234567890,
                "model": "gpt-5-mini",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_test",
                        "name": "select_choice",
                        "arguments": '{"choice": "A", "reasons": "Test reasoning"}',
                    }
                ],
            },
        )

    route.mock(side_effect=responses_handler)
    yield
