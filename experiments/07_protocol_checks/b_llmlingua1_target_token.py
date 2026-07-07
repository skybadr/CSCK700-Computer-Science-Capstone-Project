"""Exp 07b — Can LLMLingua-1's length gate be bypassed with target_token?

Exp 02/03 showed LLMLingua-1 (Qwen2.5-0.5B scorer) returns short prompts
uncompressed when driven by `rate`. Reviewer question: does forcing an
absolute budget via `target_token` change that? 20 slice prompts (5/category,
seed 42, same as Exp 02) x targets {0.5, 0.3}, rate-driven vs token-driven.
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd
import tiktoken
import torch

SEED = 42
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
DATASET = ROOT.parent.parent / "AraPromptBench_dataset.json"
ENC = tiktoken.get_encoding("cl100k_base")


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    sample = []
    for cat in data["metadata"]["categories"]:
        sample.extend(rng.sample(
            [p for p in data["prompts"] if p["category"] == cat], 5))

    from llmlingua import PromptCompressor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pc = PromptCompressor(model_name="Qwen/Qwen2.5-0.5B", device_map=device)

    rows = []
    for p in sample:
        text = p["prompt"]
        n_cl100k = len(ENC.encode(text))
        n_scorer = len(pc.tokenizer(text)["input_ids"])
        for r in [0.5, 0.3]:
            for mode in ["rate", "target_token"]:
                try:
                    if mode == "rate":
                        out = pc.compress_prompt(text, rate=r)
                    else:
                        out = pc.compress_prompt(
                            text, target_token=max(1, int(n_scorer * r)))
                    comp = out["compressed_prompt"]
                    err = ""
                except Exception as e:
                    comp, err = "", f"{type(e).__name__}: {e}"
                rows.append(dict(
                    prompt_id=p["id"], category=p["category"], target=r,
                    mode=mode, orig_cl100k=n_cl100k, orig_scorer_tok=n_scorer,
                    achieved_keep=(len(ENC.encode(comp)) / n_cl100k
                                   if comp else None),
                    error=err))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "llmlingua1_target_token.csv", index=False,
              encoding="utf-8-sig")
    print("=== Mean achieved cl100k keep: rate-driven vs target_token-driven ===")
    print(df[df.error == ""].groupby(["category", "target", "mode"])
          .achieved_keep.mean().round(2).unstack(["target", "mode"]).to_string())
    print(f"\nerrors: {(df.error != '').sum()}")
    if (df.error != "").any():
        print(df[df.error != ""][["prompt_id", "target", "mode", "error"]]
              .head(10).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
