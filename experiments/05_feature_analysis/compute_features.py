"""Experiment 05a — Feature extraction for all 500 AraPromptBench prompts.

Computes the six SDR-specified features per prompt (SDR: Feature Extraction
Module). This module is written to be reused verbatim by the APCS artefact.

Features:
  character_length        chars after Unicode NFC normalisation
  token_count             cl100k_base tokens (gpt-4o-mini tokenizer)
  word_count              whitespace-delimited orthographic words
  fragmentation_ratio     token_count / word_count (tokeniser tax per word)
  structural_complexity   count of structural markers (newlines, colons,
                          bullets/enumerations, quoted blocks «», parentheses)
  morphological_density   Farasa segments per orthographic word
  task_category           from dataset annotation

Output: results/features.csv
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent.parent / "AraPromptBench_dataset.json"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ENC = tiktoken.get_encoding("cl100k_base")

STRUCT_PATTERNS = [
    r"\n",            # line breaks
    r"[:：]",          # colons (list/definition markers)
    r"«[^»]*»",       # quoted blocks
    r"[\(\)\[\]]",    # parentheses/brackets
    r"^\s*[-•*]\s",   # bullets (multiline)
    r"^\s*\d+[\.\)]\s",  # enumerations (multiline)
]


def structural_complexity(text: str) -> int:
    n = 0
    for pat in STRUCT_PATTERNS:
        n += len(re.findall(pat, text, flags=re.MULTILINE))
    return n


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    prompts = data["prompts"]

    # Windows note: farasapy's interactive mode corrupts Arabic through the
    # process pipe (mojibake) and can crash on stdin.flush; the temp-file
    # (standalone) mode segments correctly, so prompts are batched into
    # multi-line calls (one JVM spawn per chunk) with alignment verification.
    print("Initialising Farasa segmenter (standalone mode) ...")
    from farasa.segmenter import FarasaSegmenter
    segmenter = FarasaSegmenter(interactive=False)

    flat = [" ".join(unicodedata.normalize("NFC", p["prompt"]).split())
            for p in prompts]
    seg_lines: list[str] = []
    CHUNK = 100
    for start in range(0, len(flat), CHUNK):
        chunk = flat[start:start + CHUNK]
        out = segmenter.segment("\n".join(chunk)).splitlines()
        if len(out) != len(chunk):  # alignment lost -> per-line fallback
            print(f"  chunk {start}: line mismatch ({len(out)} vs "
                  f"{len(chunk)}), falling back to per-line calls", flush=True)
            out = [segmenter.segment(line) for line in chunk]
        seg_lines.extend(out)
        print(f"  segmented {min(start+CHUNK, len(flat))}/{len(flat)}",
              flush=True)

    rows = []
    for i, p in enumerate(prompts):
        text = unicodedata.normalize("NFC", p["prompt"])
        words = text.split()
        n_words = len(words)
        n_tok = len(ENC.encode(text))
        # Farasa splits clitics/affixes with '+'; segments per word =
        # (plus signs + words) / words
        n_segments = sum(w.count("+") + 1 for w in seg_lines[i].split())
        rows.append(dict(
            prompt_id=p["id"],
            task_category=p["category"],
            subcategory=p.get("subcategory", ""),
            character_length=len(text),
            token_count=n_tok,
            word_count=n_words,
            fragmentation_ratio=round(n_tok / n_words, 4),
            structural_complexity=structural_complexity(text),
            morphological_density=round(n_segments / n_words, 4),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "features.csv", index=False, encoding="utf-8-sig")
    print("\n=== Feature summary ===")
    print(df.drop(columns=["prompt_id", "subcategory"])
          .groupby("task_category").mean(numeric_only=True).round(2).to_string())
    print(f"\nSaved: {RESULTS / 'features.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
