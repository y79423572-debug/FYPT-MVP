import pytest
import os
import csv
import tempfile
import shutil
from unittest.mock import MagicMock
from project_codex.core.term_manager import TermManager
from project_codex.core.rag_engine import RAGEngine


@pytest.fixture
def temp_workspace():
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def mock_rag_engine():
    return MagicMock(spec=RAGEngine)


def test_load_terms_valid_csv(temp_workspace, mock_rag_engine):
    csv_path = os.path.join(temp_workspace, "glossary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "translation", "remark"])
        writer.writerow(["Fireball", "火球术", "Spell"])
        writer.writerow(["Sword", "长剑", ""])

    manager = TermManager(mock_rag_engine)
    terms = manager.load_terms_from_csv(csv_path)

    assert len(terms) == 2
    assert terms[0]["original"] == "Fireball"
    assert terms[0]["translation"] == "火球术"
    assert terms[1]["remark"] == ""


def test_load_terms_missing_headers(temp_workspace, mock_rag_engine):
    csv_path = os.path.join(temp_workspace, "bad_glossary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["col1", "col2"])  # Wrong headers

    manager = TermManager(mock_rag_engine)
    with pytest.raises(ValueError, match="Missing columns"):
        manager.load_terms_from_csv(csv_path)


def test_save_terms(temp_workspace, mock_rag_engine):
    csv_path = os.path.join(temp_workspace, "output.csv")
    manager = TermManager(mock_rag_engine)

    terms = [
        {"original": "Elf", "translation": "精灵", "remark": "Race"},
        {"original": "Orc", "translation": "兽人", "remark": "Race"},
    ]

    manager.save_terms_to_csv(csv_path, terms)

    # Verify file content
    with open(csv_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Elf,精灵,Race" in content
        assert "Orc,兽人,Race" in content


def test_sync_to_rag(temp_workspace, mock_rag_engine):
    csv_path = os.path.join(temp_workspace, "glossary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "translation", "remark"])
        writer.writerow(["Mana", "法力", "Magic resource"])

    manager = TermManager(mock_rag_engine)
    manager.sync_terms_to_rag(csv_path, collection_name="test_glossary")

    # Verify RAG interactions
    mock_rag_engine.clear_collection.assert_called_once_with("test_glossary")
    mock_rag_engine.add_texts.assert_called_once()

    # Check arguments passed to add_texts
    call_args = mock_rag_engine.add_texts.call_args
    assert call_args.kwargs["collection_name"] == "test_glossary"
    assert len(call_args.kwargs["texts"]) == 1
    assert "Original: Mana" in call_args.kwargs["texts"][0]
    assert call_args.kwargs["metadatas"][0]["translation"] == "法力"
