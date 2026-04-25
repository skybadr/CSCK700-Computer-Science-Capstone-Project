"""OpenAI Chat Completions client.

Wraps the official `openai` SDK into the BaseLLMClient interface, with
per-model cost estimation in USD.
"""
from __future__ import annotations

import os
import time

from src.llm.base import BaseLLMClient, LLMResponse


# Pricing per 1M tokens, USD. Update as needed from
# https://openai.com/api/pricing
# Format: {model: (input_per_1m, output_per_1m)}
_OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":      (0.15,  0.60),
    "gpt-4o":           (2.50, 10.00),
    "gpt-4-turbo":      (10.00, 30.00),
    "gpt-3.5-turbo":    (0.50,  1.50),
}


class OpenAIClient(BaseLLMClient):
    """Thin wrapper around openai.OpenAI's chat.completions endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai is not installed. Run `pip install openai`.") from e

        self.model = model
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        start = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.perf_counter() - start

        text = completion.choices[0].message.content or ""
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

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
        pricing = _OPENAI_PRICING.get(self.model)
        if pricing is None:
            return 0.0
        in_rate, out_rate = pricing
        return (prompt_tokens / 1_000_000) * in_rate + (completion_tokens / 1_000_000) * out_rate
