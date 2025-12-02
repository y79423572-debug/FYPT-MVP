"""
Workflow Pipeline Engine for Project Codex.
Implements the multi-step localization process.
"""
import json
from typing import List, Dict, Any, Tuple
from project_codex.core.llm_client import LLMClient
from project_codex.core.rag_engine import RAGEngine
from project_codex.utils.config import ConfigManager


class WorkflowEngine:
    """
    Core Workflow Engine implementing the 4-step concurrent pipeline.
    Designed to be stateless regarding task state; state is passed in arguments.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        llm_client: LLMClient,
        rag_engine: RAGEngine,
    ):
        """
        Initialize WorkflowEngine with dependencies.

        Args:
            config_manager: Manager for prompts and keys.
            llm_client: Client for LLM calls.
            rag_engine: Engine for vector retrieval.
        """
        self.config = config_manager
        self.llm = llm_client
        self.rag = rag_engine

    def step1_extract_terms(
        self, text: str, model: str = "gpt-4-turbo"
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Step 1: Knowledge Extraction.
        Extracts proper nouns/terms, checks against glossary, returns suggestions.

        Returns:
            A tuple of (suggestions, usage_metadata).
        """
        system_prompt = self.config.get_prompt("extractor")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            response_text, metadata = self.llm.call_llm(
                model=model, messages=messages, json_mode=True
            )
            extracted_data = json.loads(response_text)

            raw_terms = []
            if isinstance(extracted_data, list):
                raw_terms = extracted_data
            elif isinstance(extracted_data, dict):
                raw_terms = extracted_data.get("terms", [])
            else:
                raw_terms = []

        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                "Failed to parse extractor response as JSON."
            ) from exc

        suggestions = []
        for item in raw_terms:
            original = item.get("original")
            if not original:
                continue

            results = self.rag.query(
                collection_name="glossary", query_text=original, n_results=1
            )

            is_known = False
            if results["documents"] and results["documents"][0]:
                stored_text = results["documents"][0][0]
                if f"Original: {original}" in stored_text:
                    is_known = True

            if not is_known:
                suggestions.append(item)

        return suggestions, metadata

    def step2_draft(
        self, text: str, model: str = "deepseek-chat"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Step 2: Drafting (P1).

        Returns:
            A tuple of (draft_text, usage_metadata).
        """
        results = self.rag.query(
            collection_name="glossary", query_text=text, n_results=10
        )

        glossary_context = ""
        if results["documents"] and results["documents"][0]:
            glossary_context = "\n".join(results["documents"][0])

        system_prompt = self.config.get_prompt("drafter_p1")
        user_content = (
            f"Relevant Glossary:\n{glossary_context}\n\n"
            f"Source Text:\n{text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        draft, metadata = self.llm.call_llm(model=model, messages=messages)
        return draft, metadata

    def step3_judge(
        self,
        draft: str,
        project_context: str,
        model: str = "gemini-1.5-pro-latest",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Step 3: Judging.

        Returns:
            A tuple of (critique_dict, usage_metadata).
        """
        system_prompt = self.config.get_prompt("judge")
        combined_system_message = (
            f"{system_prompt}\n\n"
            f"Project Context (Core Settings):\n{project_context}"
        )

        messages = [
            {"role": "system", "content": combined_system_message},
            {"role": "user", "content": f"Draft to critique:\n{draft}"},
        ]

        try:
            response_text, metadata = self.llm.call_llm(
                model=model,
                messages=messages,
                json_mode=True,
                context_caching=True,
            )
            critique = json.loads(response_text)
            return critique, metadata
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse judge response: {e}") from e

    def step4_polish(
        self,
        draft: str,
        critique: Dict[str, Any],
        model: str = "claude-3-5-sonnet-20240620",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Step 4: Polishing (P2).

        Returns:
            A tuple of (final_text, usage_metadata).
        """
        results = self.rag.query(
            collection_name="style_bank", query_text=draft, n_results=3
        )

        style_context = ""
        if results["documents"] and results["documents"][0]:
            style_context = "\n\n".join(results["documents"][0])

        system_prompt = self.config.get_prompt("polisher_p2")
        critique_text = json.dumps(critique, ensure_ascii=False)

        user_content = (
            f"Style Reference:\n{style_context}\n\n"
            f"Critique:\n{critique_text}\n\n"
            f"Draft:\n{draft}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        final_text, metadata = self.llm.call_llm(model=model, messages=messages)
        return final_text, metadata
