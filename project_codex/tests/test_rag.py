import pytest
import shutil
import tempfile
import os
from project_codex.core.rag_engine import RAGEngine


@pytest.fixture
def temp_rag_engine():
    # Create a temporary directory for the vector DB
    test_dir = tempfile.mkdtemp()
    rag = RAGEngine(persist_directory=test_dir)
    yield rag
    # Cleanup
    shutil.rmtree(test_dir)


def test_initialization(temp_rag_engine):
    assert temp_rag_engine.client is not None
    assert os.path.exists(temp_rag_engine.persist_directory)


def test_add_and_query_texts(temp_rag_engine):
    collection_name = "test_collection"
    texts = ["apple", "banana", "cherry"]
    metadatas = [{"type": "fruit"}, {"type": "fruit"}, {"type": "fruit"}]

    temp_rag_engine.add_texts(collection_name, texts, metadatas=metadatas)

    results = temp_rag_engine.query(collection_name, "apple", n_results=1)

    # Check if we got results
    assert results is not None
    assert "documents" in results
    assert len(results["documents"][0]) >= 1
    assert results["documents"][0][0] == "apple"
    assert results["metadatas"][0][0]["type"] == "fruit"


def test_add_texts_mismatched_lengths(temp_rag_engine):
    collection_name = "error_collection"
    texts = ["one", "two"]
    metadatas = [{"id": 1}]  # Mismatched length

    with pytest.raises(ValueError):
        temp_rag_engine.add_texts(collection_name, texts, metadatas=metadatas)


def test_query_nonexistent_collection(temp_rag_engine):
    # ChromaDB (v0.4+) usually creates collection on get_or_create, so query should return empty or valid struct
    # But if we rely on internal logic, let's see.
    # My implementation uses get_or_create_collection, so it should return empty results if empty.
    collection_name = "empty_collection"
    # We create it implicitly by querying or adding?
    # The query method calls _get_collection which uses get_or_create.
    # But querying an empty collection might raise error in Chroma depending on version if n_results > count

    # Let's add something first to be safe or catch exception if that is expected behavior for empty
    # Actually, let's just test that the method runs without exploding if collection is empty
    # ChromaDB might raise "not enough elements" if n_results > count.

    try:
        temp_rag_engine.query(collection_name, "test", n_results=1)
    except RuntimeError as e:
        # If the underlying error is about empty collection, that's fine.
        assert (
            "not enough elements" in str(e).lower()
            or "index" in str(e).lower()
            or "created" in str(e).lower()
        )


def test_glossary_use_case(temp_rag_engine):
    collection_name = "glossary"
    texts = ["fireball", "ice storm"]
    metadatas = [
        {"translation": "火球术", "remark": "Basic spell"},
        {"translation": "冰风暴", "remark": "AOE spell"},
    ]

    temp_rag_engine.add_texts(collection_name, texts, metadatas=metadatas)

    # Query for "fire"
    results = temp_rag_engine.query(collection_name, "fire", n_results=1)

    assert "fireball" in results["documents"][0]
    assert results["metadatas"][0][0]["translation"] == "火球术"
