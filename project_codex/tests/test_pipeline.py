import json
import pytest
from unittest.mock import MagicMock
from project_codex.core.pipeline import WorkflowEngine
from project_codex.core.llm_client import LLMClient
from project_codex.core.rag_engine import RAGEngine
from project_codex.utils.config import ConfigManager


@pytest.fixture
def mock_config():
    config = MagicMock(spec=ConfigManager)
    # We don't need get_prompt to return anything because we inject prompt in methods
    return config


@pytest.fixture
def mock_llm():
    return MagicMock(spec=LLMClient)


@pytest.fixture
def mock_rag():
    rag = MagicMock(spec=RAGEngine)
    rag.query.return_value = {"documents": [[]], "metadatas": [[]]}
    return rag


@pytest.fixture
def workflow_engine(mock_config, mock_llm, mock_rag):
    return WorkflowEngine(mock_config, mock_llm, mock_rag)


def test_step1_extract_terms(workflow_engine, mock_llm, mock_rag):
    # Mock LLM response
    mock_llm.call_llm.return_value = (
        json.dumps({"terms": [{"original": "Fireball", "type": "Spell"}]}),
        {"tokens": 10, "cost": 0.01},
    )

    # Mock RAG to return empty results (term not found)
    mock_rag.query.return_value = {"documents": [[]], "metadatas": [[]]}

    suggestions, metadata = workflow_engine.step1_extract_terms(
        "He cast Fireball.", model="test-model", system_prompt="SysPrompt"
    )

    assert len(suggestions) == 1
    assert suggestions[0]["original"] == "Fireball"
    assert metadata["tokens"] == 10
    mock_llm.call_llm.assert_called_once()
    assert mock_llm.call_llm.call_args.kwargs["json_mode"] is True


def test_step1_extract_terms_known_term(workflow_engine, mock_llm, mock_rag):
    # Mock LLM response
    mock_llm.call_llm.return_value = (
        json.dumps({"terms": [{"original": "Fireball"}]}),
        {"tokens": 10, "cost": 0.01},
    )

    # Mock RAG to return existing term
    mock_rag.query.return_value = {
        "documents": [["Original: Fireball\nTranslation: ..."]],
        "metadatas": [[{}]],
    }

    suggestions, _ = workflow_engine.step1_extract_terms(
        "He cast Fireball.", model="test-model", system_prompt="SysPrompt"
    )

    # Should be filtered out
    assert len(suggestions) == 0


def test_step2_draft(workflow_engine, mock_llm, mock_rag):
    mock_llm.call_llm.return_value = ("Draft text", {"tokens": 100, "cost": 0.1})

    # Mock RAG glossary retrieval
    mock_rag.query.return_value = {
        "documents": [["Term: A", "Term: B"]],
        "metadatas": [[{}]],
    }

    draft, metadata = workflow_engine.step2_draft(
        "Source text", model="test-model", system_prompt="SysPrompt"
    )

    assert draft == "Draft text"
    assert metadata["cost"] == 0.1
    # Verify glossary context was included
    call_args = mock_llm.call_llm.call_args
    assert "Term: A" in call_args.kwargs["messages"][1]["content"]


def test_step3_judge(workflow_engine, mock_llm):
    mock_llm.call_llm.return_value = (
        json.dumps({"issues": []}),
        {"tokens": 50, "cost": 0.05},
    )

    critique, metadata = workflow_engine.step3_judge(
        "Draft", "Context", model="test-model", system_prompt="SysPrompt"
    )

    assert critique == {"issues": []}
    assert metadata["tokens"] == 50
    assert mock_llm.call_llm.call_args.kwargs["context_caching"] is True
    assert mock_llm.call_llm.call_args.kwargs["json_mode"] is True


def test_step4_polish(workflow_engine, mock_llm, mock_rag):
    mock_llm.call_llm.return_value = ("Final text", {"tokens": 120, "cost": 0.12})

    # Mock style bank retrieval
    mock_rag.query.return_value = {"documents": [["Style Ex 1"]], "metadatas": [[{}]]}

    final, metadata = workflow_engine.step4_polish(
        "Draft", {"critique": "ok"}, model="test-model", system_prompt="SysPrompt"
    )

    assert final == "Final text"
    assert metadata["cost"] == 0.12
    call_args = mock_llm.call_llm.call_args
    assert "Style Ex 1" in call_args.kwargs["messages"][1]["content"]
