"""
Configuration Manager for Project Codex.
Handles loading and saving of configuration settings and prompts.
"""
import os
import json
from typing import Dict


DEFAULT_PROMPTS = {
    "extractor": "你是一个资深的出版编辑，请提取文中所有专有名词...",
    "drafter_p1": "你是一个通俗小说翻译家，请根据以下术语表将文本转化为流畅的英文草稿...",
    "judge": "你是一个严厉的审稿人 (Chief Editor)，请指出草稿中的逻辑错误、术语不一致和OOC（性格违规）问题...",
    "polisher_p2": "你是一个畅销书作家，请根据审稿意见和以下风格参考范例（Few-Shot），润色这段文字...",
}

DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../data/config")
PROMPTS_FILE_NAME = "system_prompts.json"


class ConfigManager:
    """
    Manages application configuration, including API keys and System Prompts.
    """

    def __init__(self, config_dir: str = DEFAULT_CONFIG_DIR):
        """
        Initialize ConfigManager.

        Args:
            config_dir: Directory to store configuration files.
        """
        self.config_dir = config_dir
        self.prompts_file_path = os.path.join(config_dir, PROMPTS_FILE_NAME)

        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)

        # Initialize prompts
        self.prompts: Dict[str, str] = {}
        self.prompts = self._load_prompts()

    def get_api_key(self, provider: str) -> str:
        """
        Retrieve API key for a specific provider from environment variables.

        Args:
            provider: The name of the provider (e.g., "OPENAI", "ANTHROPIC").

        Returns:
            The API key string.

        Raises:
            ValueError: If the API key is missing.
        """
        # Convention: Provider name + "_API_KEY" (e.g., OPENAI_API_KEY)
        key_name = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(key_name)

        if not api_key:
            raise ValueError(
                f"Missing API Key for provider '{provider}': "
                f"{key_name} is not set in environment."
            )

        return api_key

    def _load_prompts(self) -> Dict[str, str]:
        """
        Load system prompts from JSON file. If file doesn't exist, create it with defaults.

        Returns:
            Dictionary of system prompts.
        """
        if not os.path.exists(self.prompts_file_path):
            self.save_prompts(DEFAULT_PROMPTS)
            return DEFAULT_PROMPTS.copy()

        try:
            with open(self.prompts_file_path, "r", encoding="utf-8") as f:
                prompts = json.load(f)
                if not isinstance(prompts, dict):
                    raise ValueError("Prompts file is not a valid JSON object.")
                return prompts
        except Exception as e:
            raise RuntimeError(
                f"Failed to load prompts from {self.prompts_file_path}: {str(e)}"
            ) from e

    def get_prompt(self, prompt_key: str) -> str:
        """
        Get a specific system prompt by key.

        Args:
            prompt_key: The key of the prompt (e.g., "extractor", "judge").

        Returns:
            The prompt text.

        Raises:
            KeyError: If prompt key is not found.
        """
        if prompt_key not in self.prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found in configuration.")
        return self.prompts[prompt_key]

    def save_prompts(self, new_prompts: Dict[str, str]) -> None:
        """
        Save system prompts to JSON file.

        Args:
            new_prompts: Dictionary of prompts to save.
        """
        try:
            # Update internal state
            self.prompts.update(new_prompts)

            with open(self.prompts_file_path, "w", encoding="utf-8") as f:
                json.dump(self.prompts, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to save prompts to {self.prompts_file_path}: {str(e)}"
            ) from e
