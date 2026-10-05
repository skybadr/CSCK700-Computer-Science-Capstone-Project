"""LLMLingua-2 with length instructions protected (Experiment 11).

Splits the prompt into sentence-like segments, keeps every segment that
contains a length instruction verbatim, compresses the remaining text with
LLMLingua-2 at the target rate and reassembles it in the original order.
Prompts without a length instruction are compressed exactly as plain
LLMLingua-2 would compress them.
"""

from __future__ import annotations

import re

from .features import LENGTH_MARKERS

# split after sentence punctuation, colons and newlines, keeping delimiters
_SPLIT = re.compile(r"(?<=[.؟!:\n])")


def compress_protected(pc, text: str, rate: float) -> str:
    """pc: an llmlingua PromptCompressor configured for LLMLingua-2."""
    if not LENGTH_MARKERS.search(text):
        return pc.compress_prompt(text, rate=rate)["compressed_prompt"]
    out, buffer = [], []

    def flush():
        if buffer:
            chunk = "".join(buffer)
            out.append(pc.compress_prompt(chunk, rate=rate)["compressed_prompt"]
                       if chunk.strip() else chunk)
            buffer.clear()

    for seg in (s for s in _SPLIT.split(text) if s.strip()):
        if LENGTH_MARKERS.search(seg):
            flush()
            out.append(seg.strip())
        else:
            buffer.append(seg)
    flush()
    return " ".join(s for s in out if s)
