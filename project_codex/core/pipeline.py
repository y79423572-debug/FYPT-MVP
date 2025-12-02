"""
Workflow Pipeline Engine for Project Codex.
Implements the multi-step localization process.
"""
import json
from typing import List, Dict, Any
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
    ) -> List[Dict[str, str]]:
        """
        Step 1: Knowledge Extraction.
        Extracts proper nouns/terms, checks against glossary, returns suggestions.

        Args:
            text: The source text segment.
            model: The LLM model to use.

        Returns:
            List of dictionaries representing suggested terms.
        """
        system_prompt = self.config.get_prompt("extractor")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        # Call LLM with JSON mode to ensure structured output
        try:
            response_text = self.llm.call_llm(
                model=model, messages=messages, json_mode=True
            )
            extracted_data = json.loads(response_text)

            # Expected format: {"terms": [{"original": "...", "type": "..."}]}
            # Handle list directly or wrapped in dict
            raw_terms = []
            if isinstance(extracted_data, list):
                raw_terms = extracted_data
            elif isinstance(extracted_data, dict):
                raw_terms = extracted_data.get("terms", [])
            else:
                # Fallback or empty
                return []

        except (json.JSONDecodeError, ValueError) as exc:
            # Fail gracefully for extraction step? Or Fail Fast?
            # Project guide says "Fail Fast", but for non-critical extraction maybe warning?
            # Let's stick to Fail Fast for now, as bad JSON means broken pipeline step.
            raise RuntimeError(
                "Failed to parse extractor response as JSON."
            ) from exc

        suggestions = []
        for item in raw_terms:
            original = item.get("original")
            if not original:
                continue

            # Check against RAG (Glossary)
            # We assume if it exists in glossary, we don't need to suggest it (unless updating).
            # Query exact match? RAG query is semantic.
            # Ideally we check existence.
            results = self.rag.query(
                collection_name="glossary", query_text=original, n_results=1
            )

            # Simple heuristic: if distance is very small, it exists.
            # Or if the returned document matches exactly.
            is_known = False
            if results["documents"] and results["documents"][0]:
                # Check if the retrieved term's "Original" field matches.
                # RAG returns the stored text block.
                # "Original: {original}\n..."
                stored_text = results["documents"][0][0]
                if f"Original: {original}" in stored_text:
                    is_known = True

            if not is_known:
                suggestions.append(item)

        return suggestions

    def step2_draft(self, text: str, model: str = "deepseek-chat") -> str:
        """
        Step 2: Drafting (P1).
        Retrieves glossary terms and generates initial draft.

        Args:
            text: The source text segment.
            model: The LLM model to use (default DeepSeek).

        Returns:
            The generated draft text.
        """
        # 1. Retrieve relevant terms
        # Query RAG using the source text to find relevant glossary entries
        results = self.rag.query(
            collection_name="glossary", query_text=text, n_results=10
        )

        glossary_context = ""
        if results["documents"] and results["documents"][0]:
            glossary_context = "\n".join(results["documents"][0])

        # 2. Assemble Prompt
        system_prompt = self.config.get_prompt("drafter_p1")

        user_content = (
            f"Relevant Glossary:\n{glossary_context}\n\n"
            f"Source Text:\n{text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 3. Call LLM
        draft = self.llm.call_llm(model=model, messages=messages)
        return draft

    def step3_judge(
        self,
        draft: str,
        project_context: str,
        model: str = "gemini-1.5-pro-latest",
    ) -> Dict[str, Any]:
        """
        Step 3: Judging.
        Critiques the draft using context caching.

        Args:
            draft: The draft text to judge.
            project_context: Full book core settings/summary (for context caching).
            model: The LLM model (default Gemini).

        Returns:
            Structured critique (JSON).
        """
        system_prompt = self.config.get_prompt("judge")

        # We prepend the project context to be cached
        # The 'context_caching=True' flag in llm_client is a signal to the provider.
        # Efficient caching usually depends on the static prefix.

        combined_system_message = (
            f"{system_prompt}\n\n"
            f"Project Context (Core Settings):\n{project_context}"
        )

        messages = [
            {"role": "system", "content": combined_system_message},
            {"role": "user", "content": f"Draft to critique:\n{draft}"},
        ]

        try:
            response_text = self.llm.call_llm(
                model=model,
                messages=messages,
                json_mode=True,
                context_caching=True,
            )
            critique = json.loads(response_text)
            return critique
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse judge response: {e}") from e

    def step4_polish(
        self,
        draft: str,
        critique: Dict[str, Any],
        model: str = "claude-3-5-sonnet-20240620",
    ) -> str:
        """
        Step 4: Polishing (P2).
        Applies style transfer and fixes based on critique.

        Args:
            draft: The initial draft.
            critique: The critique from Step 3.
            model: The LLM model (default Claude).

        Returns:
            The final polished text.
        """
        # 1. Retrieve Style Examples (Few-Shot)
        # Query style_bank based on draft content to find similar scenes/styles
        results = self.rag.query(
            collection_name="style_bank", query_text=draft, n_results=3
        )

        style_context = ""
        if results["documents"] and results["documents"][0]:
            style_context = "\n\n".join(results["documents"][0])

        # 2. Assemble Prompt
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

        # 3. Call LLM
        final_text = self.llm.call_llm(model=model, messages=messages)
        return final_text
