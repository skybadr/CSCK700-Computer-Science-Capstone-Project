"""Wrapper around LongLLMLingua prompt compression.

LongLLMLingua extends LLMLingua for long-context scenarios. It uses
question-aware coarse-to-fine compression and dynamic compression ratios
to better preserve information relevant to a query within a long prompt.

Reference:
    Jiang, H., Wu, Q., Luo, X., Li, D., Lin, C.-Y., Yang, Y. and Qiu, L. (2024)
    'LongLLMLingua: Accelerating and enhancing LLMs in long context scenarios
    via prompt compression', ACL 2024.
    https://aclanthology.org/2024.acl-long.91/
"""
from __future__ import annotations

import time
from typing import Any

from src.compression.base import BaseCompressor, CompressionResult


class LongLLMLinguaCompressor(BaseCompressor):
    """Wrap llmlingua.PromptCompressor in LongLLMLingua mode.

    LongLLMLingua is activated by passing a question/query alongside the
    prompt context. When no question is provided it falls back to standard
    LLMLingua behaviour.
    """

    name = "longllmlingua"

    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-hf",
        device_map: str = "cpu",
    ) -> None:
        """Initialise the underlying LongLLMLingua model.

        Args:
            model_name: HF model used for token-importance scoring.
                Swap to 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' for lighter CPU runs.
            device_map: 'cpu', 'cuda', or 'auto'.
        """
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

    def compress(
        self,
        prompt: str,
        target_ratio: float = 0.5,
        question: str = "",
    ) -> CompressionResult:
        """Compress using LongLLMLingua's question-aware strategy.

        Args:
            prompt: Input prompt text (Arabic or otherwise).
            target_ratio: Fraction of tokens to keep (0.0-1.0).
            question: Optional query the prompt is meant to answer.
                      When provided, compression preserves question-relevant tokens.
        """
        start = time.perf_counter()

        kwargs: dict[str, Any] = {
            "rate": target_ratio,
            "rank_method": "longllmlingua",
        }
        if question:
            kwargs["question"] = question

        result: dict[str, Any] = self._compressor.compress_prompt(
            prompt,
            **kwargs,
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
                "question": question,
                "raw_response": {
                    k: v for k, v in result.items()
                    if k not in {"compressed_prompt"}
                },
            },
        )
