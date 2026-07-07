"""Feature extraction for Arabic prompts (APCS Feature Extraction Module).

Mirrors the definitions used throughout the AraPromptBench experiments
(Experiment 05): the same code path produced the calibration data, so
inference-time features are consistent with the rules by construction.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

_STRUCT_PATTERNS = [
    r"\n",               # line breaks
    r"[:：]",             # colons (list/definition markers)
    r"«[^»]*»",          # quoted blocks
    r"[\(\)\[\]]",       # parentheses/brackets
    r"^\s*[-•*]\s",      # bullets
    r"^\s*\d+[\.\)]\s",  # enumerations
]


@dataclass(frozen=True)
class PromptFeatures:
    character_length: int
    token_count: int
    word_count: int
    fragmentation_ratio: float
    structural_complexity: int
    morphological_density: float | None  # requires optional Farasa/Java

    def as_dict(self) -> dict:
        return asdict(self)


def structural_complexity(text: str) -> int:
    return sum(len(re.findall(p, text, flags=re.MULTILINE))
               for p in _STRUCT_PATTERNS)


def extract_features(text: str, *, morphology: bool = False) -> PromptFeatures:
    """Compute the APCS feature vector for one prompt.

    morphology=True computes Farasa morphological density (requires the
    'morphology' extra and a Java runtime; adds noticeable latency). The
    shipped decision rules do not use it (Experiment 05: weak predictor),
    so it defaults to off.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt must be a non-empty string")
    text = unicodedata.normalize("NFC", text)
    words = text.split()
    n_tok = len(_ENC.encode(text))
    density = _morphological_density(text) if morphology else None
    return PromptFeatures(
        character_length=len(text),
        token_count=n_tok,
        word_count=len(words),
        fragmentation_ratio=round(n_tok / len(words), 4),
        structural_complexity=structural_complexity(text),
        morphological_density=density,
    )


def _morphological_density(text: str) -> float:
    # standalone mode: farasapy's interactive mode corrupts Arabic through
    # the Windows process pipe (AraPromptBench Experiment 05 finding)
    from farasa.segmenter import FarasaSegmenter
    seg = FarasaSegmenter(interactive=False).segment(" ".join(text.split()))
    words = seg.split()
    return round(sum(w.count("+") + 1 for w in words) / len(words), 4)
