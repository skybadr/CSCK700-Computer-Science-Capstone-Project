"""Vertical-slice experiment.

Runs the full pipeline end-to-end on the seed prompts:
    - Loads ~20 Arabic prompts.
    - Applies the No-Op baseline, Random-Deletion baseline, and (optionally) LLMLingua.
    - Optionally calls an LLM on both original and compressed prompts.
    - Computes TCR and BERTScore.
    - Writes results to results/001_vertical_slice/results.csv.

Usage:
    python -m experiments.001_vertical_slice.run                # full run with LLM
    python -m experiments.001_vertical_slice.run --dry-run      # no LLM calls
    python -m experiments.001_vertical_slice.run --no-llmlingua # skip LLMLingua (faster)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.compression.base import BaseCompressor
from src.compression.baseline import NoOpCompressor, RandomDeletionCompressor
from src.pipeline.runner import run_experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "seed_prompts.json"
OUTPUT_PATH = REPO_ROOT / "results" / "001_vertical_slice" / "results.csv"


def build_compressors(use_llmlingua: bool) -> list[BaseCompressor]:
    compressors: list[BaseCompressor] = [
        NoOpCompressor(),
        RandomDeletionCompressor(seed=42),
    ]
    if use_llmlingua:
        # Lazy import to avoid loading torch unless we need it
        from src.compression.llmlingua_wrapper import LLMLinguaCompressor
        # TinyLlama is a much lighter scorer than the LLMLingua paper default;
        # appropriate for a CPU-bound vertical slice. Swap to a stronger model
        # when scaling up.
        compressors.append(
            LLMLinguaCompressor(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device_map="cpu",
            )
        )
    return compressors


def build_llm(dry_run: bool):
    if dry_run:
        return None

    provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY not set. Falling back to dry-run mode.")
            return None
        from src.llm.openai_client import OpenAIClient
        return OpenAIClient(model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"))

    raise ValueError(f"Unsupported LLM provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vertical-slice experiment.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls; only run compression + prompt-level BERTScore.")
    parser.add_argument("--no-llmlingua", action="store_true",
                        help="Skip LLMLingua (use baselines only). Useful for first-time setup.")
    parser.add_argument("--target-ratio", type=float, default=0.5,
                        help="Fraction of tokens to keep (0.0–1.0). Default: 0.5")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="If set, only process the first N prompts.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    compressors = build_compressors(use_llmlingua=not args.no_llmlingua)
    llm = build_llm(dry_run=args.dry_run)

    rows = run_experiment(
        prompts_path=PROMPTS_PATH,
        compressors=compressors,
        target_ratio=args.target_ratio,
        llm=llm,
        output_path=OUTPUT_PATH,
        max_prompts=args.max_prompts,
    )

    # --- Quick summary ---
    print("\n=== Summary ===")
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    for method, method_rows in by_method.items():
        n = len(method_rows)
        avg_tcr = sum(r["tcr"] for r in method_rows) / n
        avg_pf1 = sum(r.get("prompt_bertscore_f1", 0) or 0 for r in method_rows) / n
        avg_of1_vals = [r.get("output_bertscore_f1") for r in method_rows
                        if r.get("output_bertscore_f1") is not None]
        avg_of1 = sum(avg_of1_vals) / len(avg_of1_vals) if avg_of1_vals else None
        cost = sum(r.get("llm_cost_usd", 0) for r in method_rows)

        print(f"\n  {method}")
        print(f"    n             = {n}")
        print(f"    avg TCR       = {avg_tcr:.3f}")
        print(f"    avg prompt F1 = {avg_pf1:.3f}")
        if avg_of1 is not None:
            print(f"    avg output F1 = {avg_of1:.3f}")
        print(f"    total cost    = ${cost:.4f}")


if __name__ == "__main__":
    main()
