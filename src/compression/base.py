"""Abstract base class for prompt compression methods.

All compression methods (LLMLingua, LongLLMLingua, baselines, etc.) implement
this interface so they can be used interchangeably in the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompressionResult:
    """Result of compressing a single prompt.

    Attributes:
        original: The original (uncompressed) prompt text.
        compressed: The compressed prompt text.
        method: Name of the compression method used.
        target_ratio: The target compression ratio requested (e.g. 0.5).
        actual_ratio: Achieved compression ratio (compressed_tokens / original_tokens).
        original_token_count: Token count of the original prompt.
        compressed_token_count: Token count of the compressed prompt.
        latency_seconds: Wall-clock time the compression took.
        metadata: Free-form dict for method-specific extras (e.g. retained indices).
    """
    original: str
    compressed: str
    method: str
    target_ratio: float
    actual_ratio: float
    original_token_count: int
    compressed_token_count: int
    latency_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_compression_ratio(self) -> float:
        """TCR = 1 - (compressed / original). Higher = more compression."""
        if self.original_token_count == 0:
            return 0.0
        return 1.0 - (self.compressed_token_count / self.original_token_count)


class BaseCompressor(ABC):
    """Abstract interface that every compression method must implement."""

    name: str = "base"

    @abstractmethod
    def compress(self, prompt: str, target_ratio: float = 0.5) -> CompressionResult:
        """Compress the given prompt.

        Args:
            prompt: Input prompt text (Arabic or otherwise).
            target_ratio: Fraction of tokens to keep (0.0–1.0). Lower = more compression.

        Returns:
            A CompressionResult containing both texts, ratios, and metadata.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
