"""Evaluation metrics.

The core metrics for the dissertation are:

1. Token Compression Ratio (TCR): how much shorter the compressed prompt is.
2. BERTScore: semantic similarity between original and compressed prompts.
3. Output Similarity: BERTScore between LLM outputs from original vs compressed.

For Arabic, BERTScore quality depends heavily on the underlying model.
We default to a multilingual model that handles Arabic; for more rigorous
evaluation consider AraBERT-based scoring.
"""
from __future__ import annotations

from typing import Iterable


# Default model for BERTScore on Arabic.
# 'bert-base-multilingual-cased' is the safest baseline (works without HF gating).
# For better Arabic-specific quality, consider 'aubmindlab/bert-base-arabertv02'.
DEFAULT_BERTSCORE_MODEL = "bert-base-multilingual-cased"


def compute_token_compression_ratio(
    original_tokens: int,
    compressed_tokens: int,
) -> float:
    """TCR = 1 - (compressed / original).

    Higher = more compression. 0 = no compression. 1 = fully removed.
    """
    if original_tokens <= 0:
        return 0.0
    return 1.0 - (compressed_tokens / original_tokens)


def compute_bertscore(
    candidates: Iterable[str],
    references: Iterable[str],
    lang: str = "ar",
    model_type: str = DEFAULT_BERTSCORE_MODEL,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Compute BERTScore between candidate and reference texts.

    Args:
        candidates: Iterable of candidate (e.g. compressed) texts.
        references: Iterable of reference (e.g. original) texts.
        lang: Language code. 'ar' for Arabic.
        model_type: HF model name to use for scoring.
        verbose: Print progress.

    Returns:
        Dict with keys 'precision', 'recall', 'f1', each a list of floats
        with one score per (candidate, reference) pair.
    """
    try:
        from bert_score import score as bertscore_fn
    except ImportError as e:
        raise ImportError(
            "bert-score is not installed. Run `pip install bert-score`."
        ) from e

    cands = list(candidates)
    refs = list(references)
    if len(cands) != len(refs):
        raise ValueError(f"candidates ({len(cands)}) and references ({len(refs)}) must match")

    P, R, F = bertscore_fn(
        cands,
        refs,
        lang=lang,
        model_type=model_type,
        verbose=verbose,
    )

    return {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F.tolist(),
    }


def compute_output_similarity(
    output_original: str,
    output_compressed: str,
    lang: str = "ar",
    model_type: str = DEFAULT_BERTSCORE_MODEL,
) -> float:
    """Single-pair BERTScore F1 between two LLM outputs.

    Convenience wrapper for evaluating whether a compressed prompt produced
    an answer comparable to the uncompressed prompt's answer.
    """
    scores = compute_bertscore(
        [output_compressed],
        [output_original],
        lang=lang,
        model_type=model_type,
    )
    return scores["f1"][0]
