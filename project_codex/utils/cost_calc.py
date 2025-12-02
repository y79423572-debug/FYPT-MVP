"""
Cost Calculation Utilities for Project Codex.
Wraps LiteLLM's cost estimation logic.
"""
from typing import Any, Dict
from litellm import completion_cost  # type: ignore

def calculate_response_cost(completion_response: Any) -> float:
    """
    Calculate the cost of a completion response using LiteLLM.

    Args:
        completion_response: The response object from litellm.completion.

    Returns:
        The estimated cost in USD. Returns 0.0 if calculation fails.
    """
    try:
        # litellm.completion_cost handles various providers
        cost = completion_cost(completion_response=completion_response)
        return float(cost)
    except Exception:
        # Fail safe for cost calculation (non-critical)
        return 0.0

def estimate_tokens(text: str) -> int:
    """
    Simple heuristic for token estimation if needed (e.g. for input validation).
    Roughly 4 chars per token.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    return len(text) // 4
