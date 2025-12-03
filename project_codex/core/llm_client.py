"""
LLM Client for Project Codex.
Wraps LiteLLM to provide a unified interface for model interactions.
"""

from typing import List, Dict, Any, Optional, Tuple, cast
from litellm import completion, ModelResponse  # type: ignore
from project_codex.utils.cost_calc import calculate_response_cost


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
            # Hotfix for Gemini: Ensure 'gemini/' prefix for Google AI Studio models
            # to avoid DefaultCredentialsError (Vertex AI default)
            if "gemini" in model.lower() and not model.lower().startswith("gemini/") and not model.lower().startswith("vertex_ai/"):
                 model = f"gemini/{model}"

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
                # LiteLLM support for Gemini Context Caching usually implies implicit caching
                # based on content. However, specific parameters like 'ttl' might be needed
                # for strict control. We pass basic 'caching=True' which LiteLLM maps.
                # To be robust, we ensure we are using the correct flags.
                # NOTE: For Gemini, LiteLLM might expect specific config.
                # We will add 'ttl' if supported, but for now strict boolean is safe.
                kwargs["caching"] = True

            response = completion(**kwargs)
            # Cast to ModelResponse to satisfy type checker
            # In reality, litellm returns a ModelResponse object that behaves like a Pydantic model/dict
            model_response = cast(ModelResponse, response)

            # Extract content
            # Choices can be a list of Choices or StreamingChoices. We assume standard completion here.
            # We access index 0 safely.
            if not model_response.choices:
                raise ValueError("LLM returned no choices.")

            first_choice = model_response.choices[0]
            # Verify it has message and content. We access via getattr to avoid static type errors
            # with StreamingChoices which might not have 'message' in the type definition but works at runtime if not streaming.
            message = getattr(first_choice, "message", None)
            if not message or not message.content:
                raise ValueError("LLM returned empty content.")

            content = message.content

            # Calculate usage and cost
            # LiteLLM response object usually has a 'usage' field
            # We access it safely assuming it might be a dict or object access
            # ModelResponse usage is Usage class
            usage_info = getattr(model_response, "usage", None)
            total_tokens = 0
            if usage_info:
                # usage_info might be a Pydantic model or dict
                if isinstance(usage_info, dict):
                    total_tokens = usage_info.get("total_tokens", 0)
                else:
                    total_tokens = getattr(usage_info, "total_tokens", 0)

            # Calculate cost safely
            cost = calculate_response_cost(model_response)

            metadata = {
                "tokens": total_tokens,
                "cost": cost,
            }

            return content, metadata

        except Exception as e:
            # Fail Fast
            raise RuntimeError(f"LLM Call Failed for model {model}: {str(e)}") from e
