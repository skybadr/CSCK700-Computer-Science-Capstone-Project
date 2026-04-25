"""Baseline compression methods.

These are deliberately simple. Their purpose is to provide a floor against
which sophisticated methods (LLMLingua, etc.) can be compared.
"""
from __future__ import annotations

import random
import time

from src.compression.base import BaseCompressor, CompressionResult


def _approx_token_count(text: str) -> int:
    """Rough token count by whitespace splitting.

    For final evaluation we use the LLM's actual tokenizer (see metrics.py),
    but a whitespace count is fine for baselines and quick checks.
    """
    return len(text.split())


class NoOpCompressor(BaseCompressor):
    """Returns the prompt unchanged. The 'no compression' baseline."""

    name = "no_op"

    def compress(self, prompt: str, target_ratio: float = 1.0) -> CompressionResult:
        start = time.perf_counter()
        n = _approx_token_count(prompt)
        return CompressionResult(
            original=prompt,
            compressed=prompt,
            method=self.name,
            target_ratio=1.0,
            actual_ratio=1.0,
            original_token_count=n,
            compressed_token_count=n,
            latency_seconds=time.perf_counter() - start,
            metadata={},
        )


class RandomDeletionCompressor(BaseCompressor):
    """Deletes tokens uniformly at random until target_ratio is reached.

    Provides a 'naive compression' floor: any meaningful method should beat
    this on semantic fidelity at the same compression level.
    """

    name = "random_deletion"

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def compress(self, prompt: str, target_ratio: float = 0.5) -> CompressionResult:
        start = time.perf_counter()
        tokens = prompt.split()
        n_original = len(tokens)
        n_keep = max(1, int(round(n_original * target_ratio)))

        # Sample indices to keep, then re-sort to preserve original order
        keep_indices = sorted(self._rng.sample(range(n_original), n_keep))
        kept_tokens = [tokens[i] for i in keep_indices]
        compressed = " ".join(kept_tokens)

        actual = len(kept_tokens) / n_original if n_original else 1.0

        return CompressionResult(
            original=prompt,
            compressed=compressed,
            method=self.name,
            target_ratio=target_ratio,
            actual_ratio=actual,
            original_token_count=n_original,
            compressed_token_count=len(kept_tokens),
            latency_seconds=time.perf_counter() - start,
            metadata={"kept_indices": keep_indices},
        )
