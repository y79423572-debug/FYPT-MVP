"""
Configuration Manager for Project Codex.
Handles loading and saving of configuration settings, prompts, and profiles.
"""

import os
import json
import glob
from typing import Dict, List, Any, Optional

DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../data/config")
PROFILES_DIR = os.path.join(os.path.dirname(__file__), "../data/profiles")
PROMPTS_FILE_NAME = "system_prompts.json"

DEFAULT_PROMPTS = {
    "extractor": "你是一个资深的出版编辑，请提取文中所有专有名词...",
    "drafter_p1": "你是一个通俗小说翻译家，请根据以下术语表将文本转化为流畅的英文草稿...",
    "judge": "你是一个严厉的审稿人 (Chief Editor)，请指出草稿中的逻辑错误、术语不一致和OOC（性格违规）问题...",
    "polisher_p2": "你是一个畅销书作家，请根据审稿意见和以下风格参考范例（Few-Shot），润色这段文字...",
}

DEFAULT_MODELS = {
    "step_1_model": "gpt-4o",
    "step_2_model": "gpt-4o",
    "step_3_model": "gpt-4o",
    "step_4_model": "gpt-4o",
}


class ConfigManager:
    """
    Manages application configuration, including API keys, System Prompts, and Profiles.
    """

    def __init__(
        self, config_dir: str = DEFAULT_CONFIG_DIR, profiles_dir: str = PROFILES_DIR
    ):
        """
        Initialize ConfigManager.

        Args:
            config_dir: Directory to store configuration files.
            profiles_dir: Directory to store profiles.
        """
        self.config_dir = config_dir
        self.profiles_dir = profiles_dir
        self.prompts_file_path = os.path.join(config_dir, PROMPTS_FILE_NAME)

        # Ensure directories exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)

        # Initialize state
        self.prompts: Dict[str, str] = {}
        self.prompts = self._load_prompts()
        self.models: Dict[str, str] = DEFAULT_MODELS.copy()

        # Load environment keys if available
        self.api_keys: Dict[str, str] = {
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
            "gemini": os.getenv("GEMINI_API_KEY", ""),
        }

    def get_api_key(self, provider: str) -> str:
        """
        Retrieve API key for a specific provider.
        """
        key = self.api_keys.get(provider.lower())
        if not key:
            # Fallback to env var just in case it was set after init
            key = os.getenv(f"{provider.upper()}_API_KEY", "")

        if not key:
            raise ValueError(f"Missing API Key for provider '{provider}'.")
        return key

    def set_api_key(self, provider: str, key: str) -> None:
        """Sets an API key and updates the environment variable."""
        self.api_keys[provider.lower()] = key
        os.environ[f"{provider.upper()}_API_KEY"] = key

    def get_prompt(self, prompt_key: str) -> str:
        """
        Get a specific system prompt by key.
        """
        if prompt_key not in self.prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found in configuration.")
        return self.prompts[prompt_key]

    def _load_prompts(self) -> Dict[str, str]:
        """
        Load system prompts from JSON file. If file doesn't exist, create it with defaults.
        """
        if not os.path.exists(self.prompts_file_path):
            self.save_prompts(DEFAULT_PROMPTS)
            return DEFAULT_PROMPTS.copy()

        try:
            with open(self.prompts_file_path, "r", encoding="utf-8") as f:
                prompts = json.load(f)
                if not isinstance(prompts, dict):
                    # If invalid, return default but don't overwrite yet
                    return DEFAULT_PROMPTS.copy()
                # Merge with defaults to ensure all keys exist
                merged = DEFAULT_PROMPTS.copy()
                merged.update(prompts)
                return merged
        except Exception as e:
            raise RuntimeError(
                f"Failed to load prompts from {self.prompts_file_path}: {str(e)}"
            ) from e

    def save_prompts(self, new_prompts: Dict[str, str]) -> None:
        """
        Save system prompts to JSON file.
        """
        self.prompts.update(new_prompts)
        with open(self.prompts_file_path, "w", encoding="utf-8") as f:
            json.dump(self.prompts, f, indent=4, ensure_ascii=False)

    def get_full_config(self) -> Dict[str, Any]:
        """Returns the full configuration state (models + prompts)."""
        return {
            "models": self.models,
            "prompts": self.prompts,
            # API keys are typically not saved in profiles for security,
            # but for this specific "save current config" requirement we might need to decide.
            # Usually profiles should not contain secrets. I'll exclude them from profiles.
        }

    def load_full_config(self, config: Dict[str, Any]) -> None:
        """Loads configuration from a dictionary."""
        if "models" in config:
            self.models.update(config["models"])
        if "prompts" in config:
            self.prompts.update(config["prompts"])
            self.save_prompts(self.prompts)

    def list_profiles(self) -> List[str]:
        """Lists available profile names."""
        files = glob.glob(os.path.join(self.profiles_dir, "*.json"))
        return [os.path.splitext(os.path.basename(f))[0] for f in files]

    def save_profile(self, name: str) -> None:
        """Saves current configuration as a profile."""
        data = self.get_full_config()
        filepath = os.path.join(self.profiles_dir, f"{name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_profile(self, name: str) -> None:
        """Loads configuration from a profile."""
        filepath = os.path.join(self.profiles_dir, f"{name}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Profile {name} not found.")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.load_full_config(data)
