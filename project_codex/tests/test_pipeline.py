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
    config.get_prompt.return_value = "System Prompt"
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
    mock_llm.call_llm.return_value = json.dumps({
        "terms": [{"original": "Fireball", "type": "Spell"}]
    })

    # Mock RAG to return empty results (term not found)
    mock_rag.query.return_value = {"documents": [[]], "metadatas": [[]]}

    suggestions = workflow_engine.step1_extract_terms("He cast Fireball.")

    assert len(suggestions) == 1
    assert suggestions[0]["original"] == "Fireball"
    mock_llm.call_llm.assert_called_once()
    assert mock_llm.call_llm.call_args.kwargs["json_mode"] is True

def test_step1_extract_terms_known_term(workflow_engine, mock_llm, mock_rag):
    # Mock LLM response
    mock_llm.call_llm.return_value = json.dumps({
        "terms": [{"original": "Fireball"}]
    })

    # Mock RAG to return existing term
    mock_rag.query.return_value = {
        "documents": [["Original: Fireball\nTranslation: ..."]],
        "metadatas": [[{}]]
    }

    suggestions = workflow_engine.step1_extract_terms("He cast Fireball.")

    # Should be filtered out
    assert len(suggestions) == 0

def test_step2_draft(workflow_engine, mock_llm, mock_rag):
    mock_llm.call_llm.return_value = "Draft text"

    # Mock RAG glossary retrieval
    mock_rag.query.return_value = {
        "documents": [["Term: A", "Term: B"]],
        "metadatas": [[{}]]
    }

    draft = workflow_engine.step2_draft("Source text")

    assert draft == "Draft text"
    # Verify glossary context was included
    call_args = mock_llm.call_llm.call_args
    assert "Term: A" in call_args.kwargs["messages"][1]["content"]

def test_step3_judge(workflow_engine, mock_llm):
    mock_llm.call_llm.return_value = json.dumps({"issues": []})

    critique = workflow_engine.step3_judge("Draft", "Context")

    assert critique == {"issues": []}
    assert mock_llm.call_llm.call_args.kwargs["context_caching"] is True
    assert mock_llm.call_llm.call_args.kwargs["json_mode"] is True

def test_step4_polish(workflow_engine, mock_llm, mock_rag):
    mock_llm.call_llm.return_value = "Final text"

    # Mock style bank retrieval
    mock_rag.query.return_value = {
        "documents": [["Style Ex 1"]],
        "metadatas": [[{}]]
    }

    final = workflow_engine.step4_polish("Draft", {"critique": "ok"})

    assert final == "Final text"
    call_args = mock_llm.call_llm.call_args
    assert "Style Ex 1" in call_args.kwargs["messages"][1]["content"]
