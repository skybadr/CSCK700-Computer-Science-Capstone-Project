"""Prompt compression methods.

Each compressor implements the BaseCompressor interface so they can be
swapped freely in experiments.
"""
from src.compression.base import BaseCompressor, CompressionResult
from src.compression.baseline import NoOpCompressor, RandomDeletionCompressor

__all__ = [
    "BaseCompressor",
    "CompressionResult",
    "NoOpCompressor",
    "RandomDeletionCompressor",
]
