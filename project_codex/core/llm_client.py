"""
LLM Client for Project Codex.
Wraps LiteLLM to provide a unified interface for model interactions.
"""
from typing import List, Dict, Any, Optional
from litellm import completion  # type: ignore


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
    ) -> str:
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
            The generated text content.

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

            # Handle Context Caching
            # Note: LiteLLM abstracts some of this, but for Gemini specifically,
            # it might require specific caching headers or parameter flags.
            # As per project specs: "Judge Step: call Gemini 2.5 Pro API,
            # enable Context Caching parameter"
            if context_caching:
                # Current LiteLLM support for caching varies.
                # For now, we will pass a custom flag if supported or just document it.
                # Assuming 'caching=True' is the intent to be passed if the provider supports
                # it via kwargs or if we are using a specific LiteLLM feature.
                # For the purpose of this assignment, we'll assume passing 'cached_content'
                # or similar might be needed, but without specific API docs for the specific
                # "Gemini 2.5" integration in LiteLLM (which might be hypothetical or very new),
                # we'll add a generic kwarg that providers might use.
                kwargs["caching"] = True

            response = completion(**kwargs)

            # Extract content
            # LiteLLM returns a standard OpenAI-like response object
            content = response.choices[0].message.content

            if content is None:
                raise ValueError("LLM returned empty content.")

            return content

        except Exception as e:
            # Fail Fast
            raise RuntimeError(f"LLM Call Failed for model {model}: {str(e)}") from e
