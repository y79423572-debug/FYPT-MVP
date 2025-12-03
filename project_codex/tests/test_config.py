import os
import json
import pytest
import tempfile
import shutil
from project_codex.utils.config import ConfigManager, DEFAULT_PROMPTS


@pytest.fixture
def temp_config_manager():
    # Create a temporary directory for config
    test_dir = tempfile.mkdtemp()

    # Initialize ConfigManager with temp dir
    cm = ConfigManager(config_dir=test_dir)

    yield cm, test_dir

    # Cleanup
    shutil.rmtree(test_dir)


def test_initialization_creates_default_prompts(temp_config_manager):
    cm, test_dir = temp_config_manager
    prompts_file = os.path.join(test_dir, "system_prompts.json")

    assert os.path.exists(prompts_file)
    assert cm.get_prompt("extractor") == DEFAULT_PROMPTS["extractor"]


def test_get_api_key_success(monkeypatch, temp_config_manager):
    cm, _ = temp_config_manager
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "secret_key_123")

    key = cm.get_api_key("TEST_PROVIDER")
    assert key == "secret_key_123"


def test_get_api_key_missing(monkeypatch, temp_config_manager):
    cm, _ = temp_config_manager
    monkeypatch.delenv("MISSING_PROVIDER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing API Key"):
        cm.get_api_key("MISSING_PROVIDER")


def test_load_and_save_prompts(temp_config_manager):
    cm, test_dir = temp_config_manager

    new_prompts = {"test_prompt": "This is a test prompt."}
    cm.save_prompts(new_prompts)

    # Verify in memory
    assert cm.get_prompt("test_prompt") == "This is a test prompt."

    # Verify file content
    prompts_file = os.path.join(test_dir, "system_prompts.json")
    with open(prompts_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data["test_prompt"] == "This is a test prompt."
    # Should preserve defaults
    assert saved_data["extractor"] == DEFAULT_PROMPTS["extractor"]


def test_get_prompt_missing_key(temp_config_manager):
    cm, _ = temp_config_manager
    with pytest.raises(KeyError):
        cm.get_prompt("non_existent_key")


def test_load_prompts_corrupted_file(temp_config_manager):
    cm, test_dir = temp_config_manager
    prompts_file = os.path.join(test_dir, "system_prompts.json")

    # Write invalid JSON
    with open(prompts_file, "w", encoding="utf-8") as f:
        f.write("{invalid_json")

    # Re-initialize to trigger load
    with pytest.raises(RuntimeError):
        ConfigManager(config_dir=test_dir)
