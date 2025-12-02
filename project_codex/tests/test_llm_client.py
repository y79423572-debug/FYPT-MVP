import pytest
from unittest.mock import MagicMock, patch
from project_codex.core.llm_client import LLMClient

@pytest.fixture
def llm_client():
    return LLMClient()

@patch("project_codex.core.llm_client.completion_cost")
@patch("project_codex.core.llm_client.completion")
def test_call_llm_success(mock_completion, mock_cost, llm_client):
    # Mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello, world!"
    mock_response.get.return_value = {"total_tokens": 10}
    mock_completion.return_value = mock_response

    mock_cost.return_value = 0.001

    messages = [{"role": "user", "content": "Hi"}]
    content, metadata = llm_client.call_llm(model="gpt-3.5-turbo", messages=messages)

    assert content == "Hello, world!"
    assert metadata["tokens"] == 10
    assert metadata["cost"] == 0.001
    mock_completion.assert_called_once()

    # Verify args
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "gpt-3.5-turbo"
    assert call_kwargs["messages"] == messages
    assert call_kwargs["temperature"] == 0.7

@patch("project_codex.core.llm_client.completion")
def test_call_llm_failure(mock_completion, llm_client):
    mock_completion.side_effect = Exception("API Error")

    with pytest.raises(RuntimeError, match="LLM Call Failed"):
        llm_client.call_llm(model="gpt-4", messages=[])

@patch("project_codex.core.llm_client.completion")
def test_call_llm_with_context_caching(mock_completion, llm_client):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Cached"
    mock_completion.return_value = mock_response

    llm_client.call_llm(model="gemini-pro", messages=[], context_caching=True)

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs.get("caching") is True

@patch("project_codex.core.llm_client.completion")
def test_call_llm_json_mode(mock_completion, llm_client):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "{}"
    mock_completion.return_value = mock_response

    llm_client.call_llm(model="gpt-4", messages=[], json_mode=True)

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}
