"""Abstract LLM client.

We define a thin interface so that any provider (OpenAI, Anthropic, local
HF model, etc.) can be plugged into the experiment runner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Structured response from an LLM call.

    Attributes:
        text: Generated text.
        prompt_tokens: Tokens consumed by the input prompt.
        completion_tokens: Tokens generated.
        total_tokens: Sum of the above.
        model: Name of the model used.
        latency_seconds: Wall-clock latency for the call.
        cost_usd: Estimated cost in USD (0.0 if not computed).
    """
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    latency_seconds: float
    cost_usd: float = 0.0


class BaseLLMClient(ABC):
    """Abstract LLM client. Concrete subclasses wrap a specific API."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a prompt and return the response.

        Args:
            prompt: Input text.
            max_tokens: Cap on generated tokens.
            temperature: Sampling temperature. Use 0.0 for reproducibility.
        """
        ...
