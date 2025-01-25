import pytest
from unittest.mock import Mock, patch
import json
import os

from . import oai
from .oai import LLMClient, prepare_openai, post_process_response, run_cached_request, arun_cached_request

class MockOpenAIResponse:
    def dict(self):
        return {"response": "test"}

class MockRateLimitError(Exception):
    pass

class MockCompletions:
    def create(self, **kwargs):
        return MockOpenAIResponse()

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()

class MockEmbeddings:
    def create(self, **kwargs):
        return MockOpenAIResponse()

class MockModerations:
    def create(self, **kwargs):
        return MockOpenAIResponse()

class MockOpenAI:
    def __init__(self, **kwargs):
        self.default_headers = kwargs.get('default_headers', {})
        self.default_query = kwargs.get('default_query', {})
        self.chat = MockChat()
        self.embeddings = MockEmbeddings()
        self.moderations = MockModerations()
        self.RateLimitError = MockRateLimitError

def test_openai_sync():
    """Test basic OpenAI client functionality with a simple completion request"""
    mock_openai = MockOpenAI()
    client = LLMClient(
        openai=mock_openai,
        complete=mock_openai.chat.completions.create,
        embed=mock_openai.embeddings.create,
        moderation=mock_openai.moderations.create,
        RateLimitError=MockRateLimitError
    )

    response = run_cached_request(
        client=client,
        request_type="complete",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user", 
                "content": "What is 2+2?"
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=50
    )

    assert response == {"response": "test"}

@patch('openai.OpenAI')
@patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
def test_openai_headers(mock_openai):
    """Test OpenAI client with custom headers"""
    mock_instance = MockOpenAI(default_headers={"X-Custom-Header": "test", "X-Request-Source": "autoevals"})
    mock_openai.return_value = mock_instance
    with patch.dict(os.environ, {'OPENAI_DEFAULT_HEADERS': json.dumps({"X-Custom-Header": "test"})}):
        client, wrapped = prepare_openai()
        assert isinstance(client, LLMClient)
        assert mock_instance.default_headers["X-Custom-Header"] == "test"
        assert mock_instance.default_headers["X-Request-Source"] == "autoevals"

@patch('openai.OpenAI')
@patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
def test_openai_query_params(mock_openai):
    """Test OpenAI client with custom query parameters"""
    mock_instance = MockOpenAI(default_query={"custom_param": "test"})
    mock_openai.return_value = mock_instance
    with patch.dict(os.environ, {'OPENAI_DEFAULT_QUERY': json.dumps({"custom_param": "test"})}):
        client, wrapped = prepare_openai()
        assert isinstance(client, LLMClient)
        assert mock_instance.default_query["custom_param"] == "test"

@patch('openai.OpenAI')
@patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
def test_invalid_header_json(mock_openai):
    """Test handling of invalid header JSON"""
    mock_instance = MockOpenAI(default_headers={"X-Request-Source": "autoevals"})
    mock_openai.return_value = mock_instance
    with patch.dict(os.environ, {'OPENAI_DEFAULT_HEADERS': 'invalid json'}):
        client, wrapped = prepare_openai()
        assert isinstance(client, LLMClient)
        assert mock_instance.default_headers["X-Request-Source"] == "autoevals"
        assert len(mock_instance.default_headers) == 1

@patch('openai.OpenAI')
@patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
def test_invalid_query_json(mock_openai):
    """Test handling of invalid query JSON"""
    mock_instance = MockOpenAI()
    mock_openai.return_value = mock_instance
    with patch.dict(os.environ, {'OPENAI_DEFAULT_QUERY': 'invalid json'}):
        client, wrapped = prepare_openai()
        assert isinstance(client, LLMClient)
        assert len(mock_instance.default_query) == 0
