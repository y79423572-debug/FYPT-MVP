"""
Database Manager for Project Codex.
Handles SQLite operations and Circuit Breaker logic.
"""
import sqlite3
import os
from typing import Optional, List, Dict, Any

# Default DB path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "../data/tasks.db")


class GlobalCircuitBreakerError(Exception):
    """Raised when the circuit breaker is open."""


class DBManager:
    """
    Manages SQLite database for task tracking and logging.
    Also implements a simple Circuit Breaker.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize DB Manager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()

        # Circuit Breaker State (In-Memory)
        self.error_count = 0
        self.circuit_open = False
        self.max_errors = 5

    def _ensure_db_dir(self):
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _get_connection(self):
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            raise RuntimeError(
                f"Failed to connect to database at {self.db_path}: {e}"
            ) from e

    def _init_db(self):
        """Create necessary tables if they don't exist."""
        create_tasks_table = """
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            status TEXT NOT NULL, -- Pending, Processing, Done, Error
            current_step TEXT,
            error_message TEXT,
            draft_text TEXT,
            critique TEXT,
            final_text TEXT,
            token_usage INTEGER DEFAULT 0,
            cost_estimate REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(create_tasks_table)
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to initialize database tables: {e}") from e
        finally:
            conn.close()

    def create_task(self, task_id: str, file_name: str) -> None:
        """Create a new task entry."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO tasks (id, file_name, status) VALUES (?, ?, ?)",
                (task_id, file_name, "Pending"),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to create task {task_id}: {e}") from e
        finally:
            conn.close()

    def update_task_status(
        self,
        task_id: str,
        status: str,
        step: Optional[str] = None,
        error: Optional[str] = None,
        draft_text: Optional[str] = None,
        critique: Optional[str] = None,
        final_text: Optional[str] = None,
        token_usage: Optional[int] = None,
        cost_estimate: Optional[float] = None,
    ) -> None:
        """Update task status and details."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
            params: List[Any] = [status]

            if step:
                updates.append("current_step = ?")
                params.append(step)

            if error:
                updates.append("error_message = ?")
                params.append(error)

            if draft_text:
                updates.append("draft_text = ?")
                params.append(draft_text)

            if critique:
                updates.append("critique = ?")
                params.append(critique)

            if final_text:
                updates.append("final_text = ?")
                params.append(final_text)

            if token_usage is not None:
                updates.append("token_usage = ?")
                params.append(token_usage)

            if cost_estimate is not None:
                updates.append("cost_estimate = ?")
                params.append(cost_estimate)

            params.append(task_id)

            query = f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, tuple(params))
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to update task {task_id}: {e}") from e
        finally:
            conn.close()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task details."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to get task {task_id}: {e}") from e
        finally:
            conn.close()

    def record_error(self):
        """
        Record an API error. If threshold reached, open circuit.
        """
        self.error_count += 1
        if self.error_count >= self.max_errors:
            self.circuit_open = True
            raise GlobalCircuitBreakerError(
                "Circuit Breaker Triggered: Too many consecutive errors (5). Queue stopped."
            )

    def reset_circuit_breaker(self):
        """Reset the circuit breaker."""
        self.error_count = 0
        self.circuit_open = False

    def check_circuit(self):
        """Check if circuit is open and raise error if so."""
        if self.circuit_open:
            raise GlobalCircuitBreakerError("Circuit Breaker is OPEN. Operation denied.")
