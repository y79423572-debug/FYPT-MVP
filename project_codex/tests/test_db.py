import pytest
import sqlite3
import os
import tempfile
import shutil
from project_codex.utils.db import DBManager, GlobalCircuitBreakerError


@pytest.fixture
def temp_db_manager():
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test_tasks.db")
    db_manager = DBManager(db_path=db_path)
    yield db_manager
    shutil.rmtree(test_dir)


def test_db_initialization(temp_db_manager):
    assert os.path.exists(temp_db_manager.db_path)

    conn = sqlite3.connect(temp_db_manager.db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks';"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_create_and_get_task(temp_db_manager):
    task_id = "task_123"
    file_name = "novel.txt"

    temp_db_manager.create_task(task_id, file_name)

    task = temp_db_manager.get_task(task_id)
    assert task is not None
    assert task["id"] == task_id
    assert task["filename"] == file_name
    assert task["status"] == "Pending"


def test_update_task(temp_db_manager):
    task_id = "task_456"
    temp_db_manager.create_task(task_id, "doc.txt")

    temp_db_manager.update_task_status(task_id, "Processing", step="Extraction")

    task = temp_db_manager.get_task(task_id)
    assert task["status"] == "Processing"
    assert task["current_step"] == "Extraction"

    temp_db_manager.update_task_status(task_id, "Error", error_log="API Timeout")
    task_updated = temp_db_manager.get_task(task_id)
    assert task_updated["status"] == "Error"
    assert task_updated["error_log"] == "API Timeout"


def test_update_task_content(temp_db_manager):
    task_id = "task_content"
    temp_db_manager.create_task(task_id, "story.txt")

    temp_db_manager.update_task_status(
        task_id,
        "Processing",
        result_p1="Once upon a time",
        result_judge='{"score": 5}',
        result_final="Once upon a time...",
    )

    task = temp_db_manager.get_task(task_id)
    assert task["result_p1"] == "Once upon a time"
    assert task["result_judge"] == '{"score": 5}'
    assert task["result_final"] == "Once upon a time..."


def test_circuit_breaker_trigger(temp_db_manager):
    # Verify initial state
    assert temp_db_manager.error_count == 0
    assert not temp_db_manager.circuit_open

    # Record 4 errors
    for _ in range(4):
        temp_db_manager.record_error()

    assert temp_db_manager.error_count == 4
    assert not temp_db_manager.circuit_open

    # 5th error should trigger
    with pytest.raises(GlobalCircuitBreakerError):
        temp_db_manager.record_error()

    assert temp_db_manager.circuit_open

    # Subsequent check should raise
    with pytest.raises(GlobalCircuitBreakerError):
        temp_db_manager.check_circuit()


def test_circuit_breaker_reset(temp_db_manager):
    # Trigger it first
    for _ in range(5):
        try:
            temp_db_manager.record_error()
        except GlobalCircuitBreakerError:
            pass

    assert temp_db_manager.circuit_open

    # Reset
    temp_db_manager.reset_circuit_breaker()
    assert not temp_db_manager.circuit_open
    assert temp_db_manager.error_count == 0

    # Should not raise now
    temp_db_manager.check_circuit()
