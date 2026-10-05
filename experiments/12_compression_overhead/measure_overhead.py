"""Exp 12 — How long does compression itself take?

Measures wall-clock time per prompt for the APCS recommendation step and for
each compressor, on a stratified sample of 100 v2 dev prompts (25 per
category), target rate 0.5, on the project workstation:
  - APCS recommend(): v1 rules and v2 tree (CPU; feature extraction + rule)
  - LLMLingua-2 (XLM-RoBERTa-large) on GPU and on CPU
  - LLMLingua-2-protected on GPU
  - LLMLingua (Qwen2.5-0.5B scorer) on GPU
Five warm-up calls precede each timed run; GPU timings synchronise CUDA.
Output: results/overhead.csv (per prompt), results/overhead_summary.json
"""

import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(PROJECT / "apcs"))
from apcs import APCSSelector  # noqa: E402
from apcs.protect import compress_protected  # noqa: E402

SEED, RATE, PER_CAT, WARMUP = 42, 0.5, 25, 5
ENC = tiktoken.get_encoding("cl100k_base")


def sample_prompts():
    v2 = json.loads((PROJECT / "AraPromptBench_v2.json").read_text(encoding="utf-8"))
    dev = [p for p in v2["prompts"] if p["split"] == "dev"]
    rng = random.Random(SEED)
    out = []
    for cat in ("instruction", "summarisation", "qa", "creative"):
        pool = [p for p in dev if p["category"] == cat]
        out += rng.sample(pool, PER_CAT)
    return out


def timed(fn, prompts, sync=False):
    for p in prompts[:WARMUP]:
        fn(p)
    times = []
    for p in prompts:
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(p)
        if sync:
            torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - t0))
    return times


def main():
    from llmlingua import PromptCompressor
    prompts = sample_prompts()
    tokens = [len(ENC.encode(p["prompt"])) for p in prompts]
    v1, v2 = APCSSelector("v1"), APCSSelector("v2")
    runs = {
        "apcs_v1_recommend": (lambda p: v1.recommend(p["prompt"]), False),
        "apcs_v2_recommend": (lambda p: v2.recommend(p["prompt"], p["category"]), False),
    }
    ll2_gpu = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                               use_llmlingua2=True, device_map="cuda")
    runs["llmlingua2_gpu"] = (lambda p: ll2_gpu.compress_prompt(p["prompt"], rate=RATE), True)
    runs["llmlingua2_protected_gpu"] = (lambda p: compress_protected(ll2_gpu, p["prompt"], RATE), True)
    results = {k: timed(fn, prompts, sync) for k, (fn, sync) in runs.items()}
    del ll2_gpu
    torch.cuda.empty_cache()

    qwen = PromptCompressor(model_name="Qwen/Qwen2.5-0.5B", device_map="cuda")
    results["llmlingua_qwen_gpu"] = timed(lambda p: qwen.compress_prompt(p["prompt"], rate=RATE),
                                          prompts, True)
    del qwen
    torch.cuda.empty_cache()

    ll2_cpu = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                               use_llmlingua2=True, device_map="cpu")
    results["llmlingua2_cpu"] = timed(lambda p: ll2_cpu.compress_prompt(p["prompt"], rate=RATE),
                                      prompts)

    df = pd.DataFrame({"prompt_id": [p["id"] for p in prompts],
                       "category": [p["category"] for p in prompts],
                       "tokens": tokens, **results})
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "overhead.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_prompts": len(prompts), "rate": RATE, "seed": SEED,
        "mean_tokens": round(float(np.mean(tokens)), 1),
        "hardware": {"cpu": platform.processor(), "cpu_threads": torch.get_num_threads(),
                     "gpu": torch.cuda.get_device_name(0)},
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
        "ms_per_prompt": {k: {"median": round(float(np.median(v)), 2),
                              "p95": round(float(np.percentile(v, 95)), 2),
                              "mean": round(float(np.mean(v)), 2)}
                          for k, v in results.items()},
        "ms_per_1k_tokens_median": {k: round(float(np.median(np.array(v) / np.array(tokens) * 1000)), 2)
                                    for k, v in results.items()},
    }
    (RESULTS / "overhead_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
