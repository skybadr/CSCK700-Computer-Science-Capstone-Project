"""Experiment 03 — Full local compression benchmark on AraPromptBench (500 prompts).

Methods (per Experiment 02 decisions):
  - noop                 passthrough control
  - random_deletion      word deletion, per-prompt deterministic seed
  - llmlingua_qwen       LLMLingua-1, Qwen2.5-0.5B perplexity scorer
  - llmlingua2           LLMLingua-2, XLM-RoBERTa-large classifier

Grid: 500 prompts x rates {0.7, 0.5, 0.3} (+ one noop row per prompt).
Records per row: cl100k token counts, achieved keep-rate, TCR, prompt-level
BERTScore F1 (AraBERT primary + mBERT sensitivity), latency, compressed text.
No API calls — everything local. Output-level evaluation is Phase 3.

Outputs (results/):
  benchmark_results.csv   master ExperimentRow-style table (with compressed texts)
  manifest.json           reproducibility manifest (versions, seed, dataset hash)
"""

import hashlib
import json
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tiktoken
import torch

SEED = 42
RATES = [0.7, 0.5, 0.3]

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent.parent / "AraPromptBench_dataset.json"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ENC = tiktoken.get_encoding("cl100k_base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ntok(text: str) -> int:
    return len(ENC.encode(text))


def random_deletion(text: str, rate: float, prompt_id: str) -> str:
    rng = random.Random(f"{SEED}-{prompt_id}-{rate}")
    words = text.split()
    n_keep = max(1, int(round(len(words) * rate)))
    idx = sorted(rng.sample(range(len(words)), n_keep))
    return " ".join(words[i] for i in idx)


def main() -> None:
    t_start = time.time()
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    prompts = data["prompts"]
    print(f"Dataset: {data['metadata']['name']} v{data['metadata']['version']}, "
          f"{len(prompts)} prompts | Device: {DEVICE}")

    from llmlingua import PromptCompressor
    print("Loading compressors ...")
    compressors = {
        "llmlingua_qwen": PromptCompressor(
            model_name="Qwen/Qwen2.5-0.5B", device_map=DEVICE),
        "llmlingua2": PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True, device_map=DEVICE),
    }

    rows = []
    for i, p in enumerate(prompts):
        text, pid = p["prompt"], p["id"]
        orig_tok = ntok(text)
        rows.append(dict(prompt_id=pid, category=p["category"],
                         subcategory=p.get("subcategory", ""), method="noop",
                         target_rate=1.0, compressed=text, orig_tokens=orig_tok,
                         comp_tokens=orig_tok, latency_s=0.0, error=""))
        for rate in RATES:
            comp = random_deletion(text, rate, pid)
            rows.append(dict(prompt_id=pid, category=p["category"],
                             subcategory=p.get("subcategory", ""),
                             method="random_deletion", target_rate=rate,
                             compressed=comp, orig_tokens=orig_tok,
                             comp_tokens=ntok(comp), latency_s=0.0, error=""))
            for name, pc in compressors.items():
                t0 = time.time()
                try:
                    out = pc.compress_prompt(text, rate=rate)
                    comp, err = out["compressed_prompt"], ""
                except Exception:
                    comp, err = "", traceback.format_exc(limit=2).splitlines()[-1]
                rows.append(dict(prompt_id=pid, category=p["category"],
                                 subcategory=p.get("subcategory", ""),
                                 method=name, target_rate=rate, compressed=comp,
                                 orig_tokens=orig_tok,
                                 comp_tokens=ntok(comp) if comp else 0,
                                 latency_s=round(time.time() - t0, 4), error=err))
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(prompts)}] compressed "
                  f"({time.time()-t_start:.0f}s elapsed)", flush=True)

    # free compressor VRAM before scoring
    del compressors
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df["achieved_keep"] = (df.comp_tokens / df.orig_tokens).round(4)
    df["tcr"] = (1 - df.achieved_keep).round(4)
    ref_map = {p["id"]: p["prompt"] for p in prompts}
    refs = df.prompt_id.map(ref_map)

    from bert_score import score as bertscore
    ok = (df.error == "") & (df.compressed != "")
    for key, model, layers in [
            ("arabert", "aubmindlab/bert-base-arabertv02", 9),
            ("mbert", "bert-base-multilingual-cased", 9)]:
        print(f"BERTScore ({model}) on {int(ok.sum())} rows ...", flush=True)
        t0 = time.time()
        _, _, f1 = bertscore(df.loc[ok, "compressed"].tolist(),
                             refs[ok].tolist(), model_type=model,
                             num_layers=layers, batch_size=64, device=DEVICE)
        df.loc[ok, f"f1_{key}"] = f1.numpy().round(4)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

    df.to_csv(RESULTS / "benchmark_results.csv", index=False, encoding="utf-8-sig")

    import bert_score as bs_pkg
    import llmlingua as ll_pkg
    import transformers
    manifest = {
        "experiment": "03_compression_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "rates": RATES,
        "n_prompts": len(prompts),
        "dataset": {"file": DATASET.name,
                    "version": data["metadata"]["version"],
                    "md5": hashlib.md5(DATASET.read_bytes()).hexdigest()},
        "methods": {
            "noop": "passthrough",
            "random_deletion": f"per-prompt seed f'{SEED}-<id>-<rate>'",
            "llmlingua_qwen": "llmlingua PromptCompressor, Qwen/Qwen2.5-0.5B",
            "llmlingua2": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        },
        "fidelity_scorers": {
            "primary": "aubmindlab/bert-base-arabertv02 (layer 9, raw F1)",
            "sensitivity": "bert-base-multilingual-cased (layer 9, raw F1)",
        },
        "token_counting": "tiktoken cl100k_base",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "llmlingua": getattr(ll_pkg, "__version__", "unknown"),
            "bert_score": bs_pkg.__version__,
            "tiktoken": tiktoken.__version__,
        },
        "device": DEVICE + (f" ({torch.cuda.get_device_name(0)})"
                            if DEVICE == "cuda" else ""),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (RESULTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== Mean by method x target rate ===")
    print(df[ok].groupby(["method", "target_rate"])
          [["achieved_keep", "tcr", "f1_arabert", "f1_mbert", "latency_s"]]
          .mean().round(3).to_string())
    n_err = int((~ok).sum())
    print(f"\nErrors: {n_err}")
    if n_err:
        print(df[~ok].groupby(["method", "category"]).size().to_string())
    print(f"\nTotal runtime: {time.time()-t_start:.0f}s")
    print(f"Saved: {RESULTS / 'benchmark_results.csv'} + manifest.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
