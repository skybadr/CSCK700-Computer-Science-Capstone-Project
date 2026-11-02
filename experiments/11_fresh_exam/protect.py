"""LLMLingua-2 with length instructions protected.

Exp 10 found that LLMLingua-2 deletes brevity/length instructions ("answer
briefly", "in three sentences", "no more than N words") and that answers
then grow 1.3-2.4x, raising total cost. This variant splits the prompt into
sentence-like segments, keeps every segment containing a length instruction
verbatim, compresses the remaining segments with LLMLingua-2 at the target
rate, and reassembles them in the original order. Prompts with no length
instruction come out exactly as plain LLMLingua-2 would compress them.
"""

import re

# same markers as Exp 10 cost_analysis.py (the mechanism test)
LENGTH_MARKERS = re.compile(
    r"بإيجاز|موجز|جملتين|ثلاث جمل|جملة واحدة|فقرة واحدة|لا يتجاوز|كلمة")
# split after sentence punctuation, colons and newlines, keeping delimiters
_SPLIT = re.compile(r"(?<=[.؟!:\n])")


def has_length_instruction(text):
    return bool(LENGTH_MARKERS.search(text))


def compress_protected(pc, text, rate):
    """pc: an llmlingua PromptCompressor configured for LLMLingua-2."""
    if not has_length_instruction(text):
        return pc.compress_prompt(text, rate=rate)["compressed_prompt"]
    segments = [s for s in _SPLIT.split(text) if s.strip()]
    out = []
    buffer = []

    def flush():
        if buffer:
            chunk = "".join(buffer)
            out.append(pc.compress_prompt(chunk, rate=rate)["compressed_prompt"]
                       if chunk.strip() else chunk)
            buffer.clear()

    for seg in segments:
        if LENGTH_MARKERS.search(seg):
            flush()
            out.append(seg.strip())
        else:
            buffer.append(seg)
    flush()
    return " ".join(s for s in out if s)
