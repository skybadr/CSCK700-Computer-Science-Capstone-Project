"""Anthropic Claude client.

Wraps the official `anthropic` SDK into the BaseLLMClient interface, with
per-model cost estimation in USD.
"""
from __future__ import annotations

import os
import time

from src.llm.base import BaseLLMClient, LLMResponse


# Pricing per 1M tokens, USD. Update as needed from
# https://www.anthropic.com/pricing
# Format: {model: (input_per_1m, output_per_1m)}
_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":   (0.80,   4.00),
    "claude-sonnet-4-5":  (3.00,  15.00),
    "claude-opus-4-5":    (15.00, 75.00),
}


class AnthropicClient(BaseLLMClient):
    """Thin wrapper around anthropic.Anthropic's messages endpoint."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic is not installed. Run `pip install anthropic`."
            ) from e

        self.model = model
        self._client = self._anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start

        text = message.content[0].text if message.content else ""
        prompt_tokens = message.usage.input_tokens
        completion_tokens = message.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens

        cost = self._estimate_cost(prompt_tokens, completion_tokens)

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=self.model,
            latency_seconds=latency,
            cost_usd=cost,
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = _ANTHROPIC_PRICING.get(self.model)
        if pricing is None:
            return 0.0
        in_rate, out_rate = pricing
        return (prompt_tokens / 1_000_000) * in_rate + (completion_tokens / 1_000_000) * out_rate
