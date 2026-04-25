"""Wrapper around LongLLMLingua prompt compression.

LongLLMLingua extends LLMLingua for long-context scenarios. It uses a
question-aware compression strategy that prioritises tokens relevant to
the query, making it better suited to QA and retrieval-style prompts.

Reference:
    Jiang et al. (2024) LongLLMLingua, ACL 2024.
    https://aclanthology.org/2024.acl-long.91/
"""
from __future__ import annotations

import time
from typing import Any

from src.compression.base import BaseCompressor, CompressionResult


class LongLLMLinguaCompressor(BaseCompressor):

    name = "longllmlingua"

    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-hf",
        device_map: str = "cpu",
    ) -> None:
        try:
            from llmlingua import PromptCompressor
        except ImportError as e:
            raise ImportError(
                "llmlingua is not installed. Run `pip install llmlingua`."
            ) from e

        self._compressor = PromptCompressor(
            model_name=model_name,
            device_map=device_map,
        )
        self._model_name = model_name

    def compress(self, prompt: str, target_ratio: float = 0.5) -> CompressionResult:
        start = time.perf_counter()

        result: dict[str, Any] = self._compressor.compress_prompt_llmlingua2(
            prompt,
            rate=target_ratio,
            use_context_level_filter=True,
        )

        latency = time.perf_counter() - start

        compressed_text: str = result.get("compressed_prompt", "")
        n_original: int = int(result.get("origin_tokens", len(prompt.split())))
        n_compressed: int = int(result.get("compressed_tokens", len(compressed_text.split())))
        actual = n_compressed / n_original if n_original else 1.0

        return CompressionResult(
            original=prompt,
            compressed=compressed_text,
            method=self.name,
            target_ratio=target_ratio,
            actual_ratio=actual,
            original_token_count=n_original,
            compressed_token_count=n_compressed,
            latency_seconds=latency,
            metadata={
                "scorer_model": self._model_name,
                "raw_response": {k: v for k, v in result.items() if k != "compressed_prompt"},
            },
        )