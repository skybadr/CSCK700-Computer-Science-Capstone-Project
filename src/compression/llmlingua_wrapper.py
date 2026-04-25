"""Wrapper around the LLMLingua prompt compression library.

LLMLingua scores tokens by perplexity using a small language model and removes
low-importance tokens. We wrap it so it conforms to BaseCompressor and can be
swapped freely in the pipeline.

Reference:
    Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y. and Qiu, L. (2023)
    'LLMLingua: Compressing prompts for accelerated inference of large language
    models', EMNLP 2023.
    https://aclanthology.org/2023.emnlp-main.825/
"""
from __future__ import annotations

import time
from typing import Any

from src.compression.base import BaseCompressor, CompressionResult


class LLMLinguaCompressor(BaseCompressor):
    """Wrap llmlingua.PromptCompressor as a BaseCompressor."""

    name = "llmlingua"

    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-hf",
        device_map: str = "cpu",
        use_llmlingua2: bool = False,
    ) -> None:
        """Initialise the underlying llmlingua model.

        Args:
            model_name: HF model used for token-importance scoring.
                The default Llama-2-7b is heavy; consider a smaller scorer
                (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0') for CPU runs.
            device_map: 'cpu', 'cuda', 'auto', etc.
            use_llmlingua2: If True, uses the LLMLingua-2 method (different
                model/scorer; see the llmlingua docs).
        """
        # Lazy import — keeps the module importable in dry-run mode
        # even if the user hasn't installed llmlingua yet.
        try:
            from llmlingua import PromptCompressor
        except ImportError as e:
            raise ImportError(
                "llmlingua is not installed. Run `pip install llmlingua`."
            ) from e

        self._compressor = PromptCompressor(
            model_name=model_name,
            device_map=device_map,
            use_llmlingua2=use_llmlingua2,
        )
        self._model_name = model_name
        self._use_llmlingua2 = use_llmlingua2
        if use_llmlingua2:
            self.name = "llmlingua2"

    def compress(self, prompt: str, target_ratio: float = 0.5) -> CompressionResult:
        start = time.perf_counter()
        # llmlingua's `rate` parameter = fraction of tokens to keep.
        result: dict[str, Any] = self._compressor.compress_prompt(
            prompt,
            rate=target_ratio,
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
                "use_llmlingua2": self._use_llmlingua2,
                "raw_response": {
                    k: v for k, v in result.items()
                    if k not in {"compressed_prompt"}  # avoid dup
                },
            },
        )
