"""
LLM Client for Project Codex.
Wraps LiteLLM to provide a unified interface for model interactions.
"""
from typing import List, Dict, Any, Optional, Tuple
from litellm import completion, completion_cost  # type: ignore


class LLMClient:
    """
    Unified client for LLM interactions using LiteLLM.
    """

    def call_llm(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        context_caching: bool = False,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Call LLM with specified parameters.

        Args:
            model: Model name (e.g., "gpt-4", "claude-3-sonnet", "gemini-pro").
            messages: List of message dictionaries (role, content).
            temperature: Sampling temperature.
            context_caching: Whether to enable context caching (Provider specific).
            max_tokens: Max tokens to generate.
            json_mode: If True, enforces JSON output (provider dependent).

        Returns:
            A tuple containing:
            - The generated text content (str).
            - A metadata dictionary containing 'tokens' (int) and 'cost' (float).

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            # Prepare kwargs
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            if context_caching:
                kwargs["caching"] = True

            response = completion(**kwargs)

            # Extract content
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty content.")

            # Calculate usage and cost
            # LiteLLM response object usually has a 'usage' field (prompt_tokens, completion_tokens, total_tokens)
            usage_info = response.get("usage", {})
            total_tokens = usage_info.get("total_tokens", 0)

            # Calculate cost safely
            try:
                cost = completion_cost(completion_response=response)
            except Exception:
                # If cost calculation fails (e.g., unknown model), default to 0.0
                cost = 0.0

            metadata = {
                "tokens": total_tokens,
                "cost": cost,
            }

            return content, metadata

        except Exception as e:
            # Fail Fast
            raise RuntimeError(f"LLM Call Failed for model {model}: {str(e)}") from e
