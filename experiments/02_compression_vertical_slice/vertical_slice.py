"""Experiment 02 — Compression vertical slice on Arabic prompts.

Verifies that the compression methods planned for the benchmark actually work
on Arabic, before building the full 500-prompt pipeline. Methods:

  - noop                      passthrough control
  - random_deletion           seed-42 word deletion (baseline)
  - llmlingua_gpt2            LLMLingua, English GPT-2 perplexity scorer (default-style)
  - llmlingua_qwen            LLMLingua, Qwen2.5-0.5B multilingual scorer (Arabic-capable)
                              (BLOOM-560m was tried first but is incompatible with
                              llmlingua: BloomConfig lacks max_position_embeddings)
  - llmlingua2                LLMLingua-2, XLM-RoBERTa-large classifier (multilingual)

20 prompts (5/category, seed 42) x target rates {0.7, 0.5, 0.3}.
Records: achieved token counts (cl100k_base), achieved keep-rate, prompt-level
BERTScore F1 under AraBERT (primary) and mBERT (sensitivity), latency, errors,
and the compressed texts themselves for qualitative inspection.
"""

import json
import random
import time
import traceback
from pathlib import Path

import pandas as pd
import tiktoken
import torch

SEED = 42
SAMPLE_PER_CATEGORY = 5
RATES = [0.7, 0.5, 0.3]

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent.parent / "AraPromptBench_dataset.json"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ENC = tiktoken.get_encoding("cl100k_base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ntok(text: str) -> int:
    return len(ENC.encode(text))


def random_deletion(text: str, rate: float, rng: random.Random) -> str:
    words = text.split()
    n_keep = max(1, int(round(len(words) * rate)))
    idx = sorted(rng.sample(range(len(words)), n_keep))
    return " ".join(words[i] for i in idx)


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    sample = []
    for cat in data["metadata"]["categories"]:
        cat_prompts = [p for p in data["prompts"] if p["category"] == cat]
        sample.extend(rng.sample(cat_prompts, SAMPLE_PER_CATEGORY))
    print(f"Sampled {len(sample)} prompts ({SAMPLE_PER_CATEGORY}/category), seed {SEED}")
    print(f"Device: {DEVICE}")

    from llmlingua import PromptCompressor

    print("\nLoading compressors ...")
    compressors = {}
    t0 = time.time()
    compressors["llmlingua_gpt2"] = PromptCompressor(
        model_name="openai-community/gpt2", device_map=DEVICE)
    print(f"  gpt2 loaded ({time.time()-t0:.0f}s)")
    t0 = time.time()
    compressors["llmlingua_qwen"] = PromptCompressor(
        model_name="Qwen/Qwen2.5-0.5B", device_map=DEVICE)
    print(f"  Qwen2.5-0.5B loaded ({time.time()-t0:.0f}s)")
    t0 = time.time()
    compressors["llmlingua2"] = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True, device_map=DEVICE)
    print(f"  llmlingua-2 loaded ({time.time()-t0:.0f}s)")

    del_rng = random.Random(SEED)
    rows = []
    for i, p in enumerate(sample):
        text = p["prompt"]
        orig_tok = ntok(text)
        # noop control (rate irrelevant, run once)
        rows.append(dict(prompt_id=p["id"], category=p["category"], method="noop",
                         target_rate=1.0, compressed=text, orig_tokens=orig_tok,
                         comp_tokens=orig_tok, latency_s=0.0, error=""))
        for rate in RATES:
            # random deletion baseline
            t0 = time.time()
            comp = random_deletion(text, rate, del_rng)
            rows.append(dict(prompt_id=p["id"], category=p["category"],
                             method="random_deletion", target_rate=rate,
                             compressed=comp, orig_tokens=orig_tok,
                             comp_tokens=ntok(comp), latency_s=time.time()-t0,
                             error=""))
            # llmlingua family
            for name, pc in compressors.items():
                t0 = time.time()
                try:
                    out = pc.compress_prompt(text, rate=rate)
                    comp = out["compressed_prompt"]
                    err = ""
                except Exception:
                    comp, err = "", traceback.format_exc(limit=2).splitlines()[-1]
                rows.append(dict(prompt_id=p["id"], category=p["category"],
                                 method=name, target_rate=rate, compressed=comp,
                                 orig_tokens=orig_tok,
                                 comp_tokens=ntok(comp) if comp else 0,
                                 latency_s=time.time()-t0, error=err))
        print(f"  [{i+1}/{len(sample)}] {p['id']} done")

    df = pd.DataFrame(rows)
    df["achieved_keep"] = df.comp_tokens / df.orig_tokens
    df["reference"] = df.prompt_id.map({p["id"]: p["prompt"] for p in sample})

    # prompt-level fidelity, both scorers, single batch each
    from bert_score import score as bertscore
    ok = (df.error == "") & (df.compressed != "")
    for key, (model, layers) in {
            "arabert": ("aubmindlab/bert-base-arabertv02", 9),
            "mbert": ("bert-base-multilingual-cased", 9)}.items():
        print(f"\nBERTScore with {model} ...")
        _, _, f1 = bertscore(df.loc[ok, "compressed"].tolist(),
                             df.loc[ok, "reference"].tolist(),
                             model_type=model, num_layers=layers,
                             batch_size=64, device=DEVICE)
        df.loc[ok, f"f1_{key}"] = f1.numpy()

    out = df.drop(columns=["reference"])
    out.to_csv(RESULTS / "slice_results.csv", index=False, encoding="utf-8-sig")

    print("\n=== Mean achieved keep-rate vs target (should track diagonally) ===")
    print(df[ok | (df.method == "noop")].groupby(["method", "target_rate"])
          [["achieved_keep", "f1_arabert", "f1_mbert", "latency_s"]]
          .mean().round(3).to_string())
    n_err = (~ok & (df.method != "noop")).sum()
    print(f"\nErrors: {n_err}")
    if n_err:
        print(df[df.error != ""][["prompt_id", "method", "target_rate", "error"]]
              .to_string(index=False))
    print(f"\nSaved: {RESULTS / 'slice_results.csv'}")


if __name__ == "__main__":
    main()
