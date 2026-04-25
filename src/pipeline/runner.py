"""End-to-end experiment runner.

For each (prompt, compressor) pair:
    1. Compress the prompt.
    2. (Optional) Send both original and compressed to an LLM.
    3. Compute metrics: TCR, prompt-level BERTScore, output-level BERTScore.
    4. Append a result row.

Designed to be called from experiment scripts in `experiments/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.compression.base import BaseCompressor
from src.evaluation.metrics import compute_bertscore, compute_output_similarity
from src.llm.base import BaseLLMClient
from src.utils.io import load_prompts, write_results_csv


def run_experiment(
    prompts_path: str | Path,
    compressors: list[BaseCompressor],
    target_ratio: float = 0.5,
    llm: BaseLLMClient | None = None,
    output_path: str | Path | None = None,
    bertscore_model: str = "bert-base-multilingual-cased",
    max_prompts: int | None = None,
) -> list[dict[str, Any]]:
    """Run a benchmark experiment.

    Args:
        prompts_path: Path to AraPromptBench-format JSON.
        compressors: List of compressor instances to evaluate.
        target_ratio: Target compression ratio (fraction of tokens to keep).
        llm: Optional LLM client. If None, runs in dry mode (no API calls).
        output_path: If given, results are written as CSV here.
        bertscore_model: HF model used for BERTScore.
        max_prompts: If given, only the first N prompts are processed.

    Returns:
        A list of result-row dicts (one per prompt-compressor pair).
    """
    data = load_prompts(prompts_path)
    prompts = data["prompts"]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    rows: list[dict[str, Any]] = []

    print(f"Running {len(prompts)} prompts × {len(compressors)} compressor(s)")
    print(f"  target_ratio = {target_ratio}")
    print(f"  llm          = {llm.__class__.__name__ if llm else 'DRY-RUN (no LLM calls)'}")
    print(f"  output       = {output_path}\n")

    # --- Phase 1: compression + LLM calls ---
    pairs = []  # (row_index, original_text, compressed_text)
    for prompt_obj in tqdm(prompts, desc="Compressing"):
        prompt_id = prompt_obj["id"]
        category = prompt_obj["category"]
        original = prompt_obj["prompt"]

        for compressor in compressors:
            cr = compressor.compress(original, target_ratio=target_ratio)

            row: dict[str, Any] = {
                "prompt_id": prompt_id,
                "category": category,
                "method": cr.method,
                "target_ratio": cr.target_ratio,
                "actual_ratio": round(cr.actual_ratio, 4),
                "tcr": round(cr.token_compression_ratio, 4),
                "original_tokens": cr.original_token_count,
                "compressed_tokens": cr.compressed_token_count,
                "compression_latency_s": round(cr.latency_seconds, 3),
                "original_text": cr.original,
                "compressed_text": cr.compressed,
                "output_original": "",
                "output_compressed": "",
                "llm_cost_usd": 0.0,
                "llm_latency_s": 0.0,
            }

            if llm is not None:
                resp_orig = llm.generate(cr.original)
                resp_comp = llm.generate(cr.compressed) if cr.compressed.strip() else None
                row["output_original"] = resp_orig.text
                row["output_compressed"] = resp_comp.text if resp_comp else ""
                row["llm_cost_usd"] = round(
                    resp_orig.cost_usd + (resp_comp.cost_usd if resp_comp else 0.0), 6
                )
                row["llm_latency_s"] = round(
                    resp_orig.latency_seconds + (resp_comp.latency_seconds if resp_comp else 0.0), 3
                )

            rows.append(row)
            pairs.append((len(rows) - 1, cr.original, cr.compressed))

    # --- Phase 2: BERTScore (batched for speed) ---
    if pairs:
        print("\nComputing prompt-level BERTScore...")
        cands = [p[2] if p[2].strip() else " " for p in pairs]  # bert-score dislikes empty strs
        refs = [p[1] for p in pairs]
        scores = compute_bertscore(cands, refs, lang="ar", model_type=bertscore_model)
        for (idx, _, _), f1 in zip(pairs, scores["f1"]):
            rows[idx]["prompt_bertscore_f1"] = round(float(f1), 4)

        if llm is not None:
            print("Computing output-level BERTScore...")
            out_cands = [r["output_compressed"] if r["output_compressed"].strip() else " "
                         for r in rows]
            out_refs = [r["output_original"] if r["output_original"].strip() else " "
                        for r in rows]
            out_scores = compute_bertscore(out_cands, out_refs, lang="ar", model_type=bertscore_model)
            for r, f1 in zip(rows, out_scores["f1"]):
                r["output_bertscore_f1"] = round(float(f1), 4)
        else:
            for r in rows:
                r["output_bertscore_f1"] = None

    # --- Phase 3: write CSV ---
    if output_path is not None:
        write_results_csv(rows, output_path)
        print(f"\nWrote {len(rows)} rows to {output_path}")

    return rows
