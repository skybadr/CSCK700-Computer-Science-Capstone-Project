"""Anthropic Claude client."""
from __future__ import annotations

import os
import time

from src.llm.base import BaseLLMClient, LLMResponse

_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80,  4.00),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-opus-4-7":           (15.00, 75.00),
}


class AnthropicClient(BaseLLMClient):

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic as anthropic_sdk
            self._sdk = anthropic_sdk
        except ImportError as e:
            raise ImportError("anthropic is not installed. Run `pip install anthropic`.") from e

        self.model = model
        self._client = self._sdk.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
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

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self.model,
            latency_seconds=latency,
            cost_usd=self._estimate_cost(prompt_tokens, completion_tokens),
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = _ANTHROPIC_PRICING.get(self.model)
        if pricing is None:
            return 0.0
        in_rate, out_rate = pricing
        return (prompt_tokens / 1_000_000) * in_rate + (completion_tokens / 1_000_000) * out_rate