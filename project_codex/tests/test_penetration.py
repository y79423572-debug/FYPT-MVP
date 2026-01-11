import pytest
import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, ANY
from project_codex.core.pipeline import WorkflowEngine, run_pipeline
from project_codex.core.llm_client import LLMClient
from project_codex.core.rag_engine import RAGEngine
from project_codex.utils.db import DBManager
from project_codex.utils.config import ConfigManager

class TestCoreLogicPenetration:

    @pytest.fixture
    def setup(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test.db")
        self.db_manager = DBManager(db_path=self.db_path)

        self.config_manager = MagicMock(spec=ConfigManager)
        self.llm_client = MagicMock(spec=LLMClient)
        self.rag_engine = MagicMock(spec=RAGEngine)

        self.workflow_engine = WorkflowEngine(
            self.config_manager, self.llm_client, self.rag_engine
        )

        yield
        shutil.rmtree(self.test_dir)

    def test_glossary_fail_safe(self, setup):
        # Setup: RAG query raises Exception (simulating empty/missing collection)
        self.rag_engine.query.side_effect = RuntimeError("Collection not found")

        # Test Step 1
        self.llm_client.call_llm.return_value = (
            json.dumps({"terms": [{"original": "Test"}]}), {"tokens": 10, "cost": 0.0}
        )

        suggestions, _ = self.workflow_engine.step1_extract_terms(
            "text", model="gpt-4", system_prompt="prompt"
        )

        # Should not crash, and treat term as unknown (suggestion returned)
        assert len(suggestions) == 1
        assert suggestions[0]["original"] == "Test"

    def test_config_penetration(self, setup):
        # Setup: Config with specific models
        config = {
            "models": {
                "step_1_model": "model-step1",
                "step_2_model": "model-step2",
                "step_3_model": "model-step3",
                "step_4_model": "model-step4"
            },
            "prompts": {
                "extractor": "prompt-1",
                "drafter_p1": "prompt-2",
                "judge": "prompt-3",
                "polisher_p2": "prompt-4"
            }
        }

        # Test run_pipeline logic passing these to engine
        task_id = "task_test"
        self.db_manager.create_task(task_id, "file.txt")

        # Mock engine methods to verify calls
        self.workflow_engine.step1_extract_terms = MagicMock(return_value=([], {}))
        self.workflow_engine.step2_draft = MagicMock(return_value=("draft", {}))
        self.workflow_engine.step3_judge = MagicMock(return_value=({}, {}))
        self.workflow_engine.step4_polish = MagicMock(return_value=("final", {}))

        run_pipeline(
            task_id, "source", config, self.db_manager, self.workflow_engine
        )

        # Verify step 1 called with specific model
        self.workflow_engine.step1_extract_terms.assert_called_with(
            "source", model="model-step1", system_prompt="prompt-1"
        )
        # Verify step 4 called with specific model
        self.workflow_engine.step4_polish.assert_called_with(
            "draft", {}, model="model-step4", system_prompt="prompt-4", style_tag="general"
        )

    def test_human_edits_persistence(self, setup):
        task_id = "task_edit"
        self.db_manager.create_task(task_id, "file.txt")

        # 1. Simulate AI Output
        self.db_manager.update_task_status(
            task_id, "Done", result_final="AI Generated Text"
        )

        # 2. Simulate User Save (Update DB)
        user_edited_text = "Human Edited Text"
        self.db_manager.update_task_status(
            task_id, "Done", result_final=user_edited_text
        )

        # 3. Retrieve for Export
        task = self.db_manager.get_task(task_id)
        assert task["result_final"] == user_edited_text
