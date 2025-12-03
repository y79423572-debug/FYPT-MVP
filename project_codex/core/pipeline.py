"""
Workflow Pipeline Engine for Project Codex.
Implements the multi-step localization process.
"""

import json
import time
from typing import List, Dict, Any, Tuple
from project_codex.core.llm_client import LLMClient
from project_codex.core.rag_engine import RAGEngine
from project_codex.utils.config import ConfigManager
from project_codex.utils.db import DBManager, GlobalCircuitBreakerError


class WorkflowEngine:
    """
    Core Workflow Engine implementing the 4-step concurrent pipeline.
    Stateless regarding task state; accepts context via arguments.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        llm_client: LLMClient,
        rag_engine: RAGEngine,
    ):
        """
        Initialize WorkflowEngine with dependencies.
        """
        self.config_manager = config_manager
        self.llm = llm_client
        self.rag = rag_engine

    def step1_extract_terms(
        self, text: str, model: str, system_prompt: str
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Step 1: Knowledge Extraction.
        """
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

        except (json.JSONDecodeError, ValueError):
            # Fallback or empty if JSON fails
            raw_terms = []
            metadata = {
                "tokens": 0,
                "cost": 0.0,
            }  # Should ideally be actual usage if call succeeded

        suggestions = []
        for item in raw_terms:
            original = item.get("original")
            if not original:
                continue

            try:
                results = self.rag.query(
                    collection_name="glossary", query_text=original, n_results=1
                )

                is_known = False
                if results["documents"] and results["documents"][0]:
                    stored_text = results["documents"][0][0]
                    if f"Original: {original}" in stored_text:
                        is_known = True
            except Exception:
                # Fail-safe: If RAG query fails (e.g. empty collection), treat as unknown
                is_known = False

            if not is_known:
                suggestions.append(item)

        return suggestions, metadata

    def step2_draft(
        self, text: str, model: str, system_prompt: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Step 2: Drafting (P1).
        """
        try:
            results = self.rag.query(
                collection_name="glossary", query_text=text, n_results=10
            )

            glossary_context = ""
            if results["documents"] and results["documents"][0]:
                glossary_context = "\n".join(results["documents"][0])
        except Exception:
            # Fail-safe
            glossary_context = ""

        # Inject glossary into prompt if placeholder exists, else append
        if "{glossary_text}" in system_prompt:
            final_system_prompt = system_prompt.replace(
                "{glossary_text}", glossary_context
            )
            user_content = f"Source Text:\n{text}"
        else:
            final_system_prompt = system_prompt
            user_content = (
                f"Relevant Glossary:\n{glossary_context}\n\n" f"Source Text:\n{text}"
            )

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": user_content},
        ]

        draft, metadata = self.llm.call_llm(model=model, messages=messages)
        return draft, metadata

    def step3_judge(
        self, draft: str, project_context: str, model: str, system_prompt: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Step 3: Judging.
        """
        if "{project_context}" in system_prompt:
            final_system_prompt = system_prompt.replace(
                "{project_context}", project_context
            )
        else:
            final_system_prompt = (
                f"{system_prompt}\n\n"
                f"Project Context (Core Settings):\n{project_context}"
            )

        messages = [
            {"role": "system", "content": final_system_prompt},
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
            # Return error dict if parsing fails
            return {
                "error": f"Failed to parse judge response: {e}",
                "raw": response_text,
            }, metadata

    def step4_polish(
        self,
        draft: str,
        critique: Dict[str, Any],
        model: str,
        system_prompt: str,
        style_tag: str = "general",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Step 4: Polishing (P2).
        """
        # Metadata Filtering
        where_filter = {"tag": style_tag} if style_tag != "general" else None

        try:
            results = self.rag.query(
                collection_name="style_bank",
                query_text=draft,
                n_results=3,
                where=where_filter
            )

            style_context = ""
            if results["documents"] and results["documents"][0]:
                style_context = "\n\n".join(results["documents"][0])
        except Exception:
            # Fail-safe
            style_context = ""

        critique_text = json.dumps(critique, ensure_ascii=False)

        # Replace placeholders
        final_system_prompt = system_prompt
        if "{critique}" in final_system_prompt:
            final_system_prompt = final_system_prompt.replace(
                "{critique}", critique_text
            )
        if "{style_examples}" in final_system_prompt:
            final_system_prompt = final_system_prompt.replace(
                "{style_examples}", style_context
            )

        # If placeholders weren't used, we might want to append context or rely on user prompt design.
        # For safety/backward compat:
        if (
            "{critique}" not in system_prompt
            and "{style_examples}" not in system_prompt
        ):
            user_content = (
                f"Style Reference:\n{style_context}\n\n"
                f"Critique:\n{critique_text}\n\n"
                f"Draft:\n{draft}"
            )
        else:
            user_content = f"Draft:\n{draft}"

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": user_content},
        ]

        final_text, metadata = self.llm.call_llm(model=model, messages=messages)
        return final_text, metadata


def run_pipeline(
    task_id: str,
    text: str,
    config: Dict[str, Any],
    db_manager: DBManager,
    workflow_engine: WorkflowEngine,
    project_context: str = "",
    style_tag: str = "general",
):
    """
    Background worker to run the full pipeline.
    """
    total_tokens = 0
    total_cost = 0.0

    # Extract config
    models = config.get("models", {})
    prompts = config.get("prompts", {})

    def update_metrics(metadata):
        nonlocal total_tokens, total_cost
        if metadata:
            total_tokens += metadata.get("tokens", 0)
            total_cost += metadata.get("cost", 0.0)

    try:
        # Helper to check cancellation
        def check_cancellation():
            current_task = db_manager.get_task(task_id)
            if current_task and current_task["status"] == "Cancelled":
                raise InterruptedError("Pipeline stopped by user.")

        # Check Circuit Breaker
        db_manager.check_circuit()

        db_manager.update_task_status(task_id, "Processing", step="Extraction")
        check_cancellation()

        # Step 1: Extraction
        _, meta1 = workflow_engine.step1_extract_terms(
            text,
            model=models.get("step_1_model", "gpt-4o"),
            system_prompt=prompts.get("extractor", ""),
        )
        update_metrics(meta1)

        db_manager.update_task_status(task_id, "Processing", step="Drafting")
        check_cancellation()

        # Step 2: Drafting
        draft, meta2 = workflow_engine.step2_draft(
            text,
            model=models.get("step_2_model", "gpt-4o"),
            system_prompt=prompts.get("drafter_p1", ""),
        )
        update_metrics(meta2)
        db_manager.update_task_status(
            task_id, "Processing", step="Judging", result_p1=draft
        )
        check_cancellation()

        # Step 3: Judging
        critique, meta3 = workflow_engine.step3_judge(
            draft,
            project_context=project_context,
            model=models.get("step_3_model", "gpt-4o"),
            system_prompt=prompts.get("judge", ""),
        )
        update_metrics(meta3)
        db_manager.update_task_status(
            task_id,
            "Processing",
            step="Polishing",
            result_judge=json.dumps(critique, ensure_ascii=False),
        )
        check_cancellation()

        # Step 4: Polishing
        final, meta4 = workflow_engine.step4_polish(
            draft,
            critique,
            style_tag=style_tag,
            model=models.get("step_4_model", "gpt-4o"),
            system_prompt=prompts.get("polisher_p2", ""),
        )
        update_metrics(meta4)

        # Final Update with metrics
        db_manager.update_task_status(
            task_id,
            "Done",
            step="Completed",
            result_final=final,
            token_usage=total_tokens,
            cost_estimate=total_cost,
        )

    except InterruptedError:
        # Task was cancelled
        pass
    except GlobalCircuitBreakerError as e:
        db_manager.update_task_status(task_id, "Error", error_log=str(e))
    except Exception as e:
        # Record error in circuit breaker
        try:
            db_manager.record_error()
        except GlobalCircuitBreakerError:
            pass  # Already recorded
        db_manager.update_task_status(task_id, "Error", error_log=str(e))
