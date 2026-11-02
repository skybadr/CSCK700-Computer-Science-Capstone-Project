"""Exp 07a — Repeat-call ceiling: the noise floor of output-level F1.

Calls gpt-4o-mini a SECOND time on each of the 500 original prompts (same
protocol as Exp 04: temp 0.0, max_tokens 512) and scores BERTScore F1
(AraBERT) between the two responses to the identical prompt. Even at
temperature 0 the API is not perfectly deterministic; this distribution is
the ceiling against which the fidelity threshold tau must be interpreted
(Exp 04 caveat / Exp 06 limitation).

Outputs: results/repeat_responses.jsonl (checkpoint), results/ceiling.csv,
printed distribution + tau recommendations.
"""

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

MODEL, TEMPERATURE, MAX_TOKENS, CONCURRENCY = "gpt-4o-mini", 0.0, 512, 8

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CKPT = RESULTS / "repeat_responses.jsonl"
EXP04 = ROOT.parent / "04_output_eval" / "results"
DATASET = ROOT.parent.parent / "AraPromptBench_dataset.json"


def call_key(pid: str, text: str) -> str:
    return hashlib.md5(f"{pid}|{text}".encode("utf-8")).hexdigest()


async def fetch(calls):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    n, t0 = 0, time.time()

    async def one(c):
        nonlocal n
        async with sem:
            for attempt in range(6):
                try:
                    r = await client.chat.completions.create(
                        model=MODEL, temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        messages=[{"role": "user", "content": c["text"]}])
                    rec = {"prompt_id": c["prompt_id"],
                           "response": r.choices[0].message.content or "",
                           "prompt_tokens": r.usage.prompt_tokens,
                           "completion_tokens": r.usage.completion_tokens,
                           "finish_reason": r.choices[0].finish_reason}
                    async with lock:
                        with CKPT.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                        if n % 100 == 0:
                            print(f"  {n}/{len(calls)} "
                                  f"({n/(time.time()-t0):.1f}/s)", flush=True)
                    return
                except Exception as e:
                    await asyncio.sleep(min(2 ** attempt * 2, 60))
            raise RuntimeError(f"retries exhausted for {c['prompt_id']}")

    await asyncio.gather(*(one(c) for c in calls))


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    prompts = {p["id"]: p["prompt"] for p in data["prompts"]}

    # first responses from Exp 04
    first = {}
    with (EXP04 / "responses.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            first[r["key"]] = r["response"]
    r1 = {pid: first[call_key(pid, text)] for pid, text in prompts.items()}

    done = set()
    if CKPT.exists():
        with CKPT.open(encoding="utf-8") as f:
            done = {json.loads(l)["prompt_id"] for l in f if l.strip()}
    todo = [{"prompt_id": pid, "text": text}
            for pid, text in prompts.items() if pid not in done]
    print(f"repeat calls: {len(prompts)} total, {len(todo)} to fetch")
    if todo:
        asyncio.run(fetch(todo))

    r2 = {}
    with CKPT.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            r2[rec["prompt_id"]] = rec["response"]

    ids = list(prompts)
    import torch
    from bert_score import score as bertscore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("scoring r1 vs r2 (AraBERT) ...")
    _, _, f1 = bertscore([r2[i] for i in ids], [r1[i] for i in ids],
                         model_type="aubmindlab/bert-base-arabertv02",
                         num_layers=9, batch_size=32, device=device)
    cat = {p["id"]: p["category"] for p in data["prompts"]}
    df = pd.DataFrame({"prompt_id": ids,
                       "category": [cat[i] for i in ids],
                       "identical_response": [r1[i] == r2[i] for i in ids],
                       "ceiling_f1_arabert": f1.numpy().round(4)})
    df.to_csv(RESULTS / "ceiling.csv", index=False, encoding="utf-8-sig")

    s = df.ceiling_f1_arabert
    print("\n=== Repeat-call ceiling, F1-AraBERT (same prompt, two calls) ===")
    print(f"identical responses : {df.identical_response.mean()*100:.1f}%")
    print(f"mean {s.mean():.4f} | median {s.median():.4f} | "
          f"p25 {s.quantile(.25):.4f} | p10 {s.quantile(.10):.4f} | "
          f"p5 {s.quantile(.05):.4f} | min {s.min():.4f}")
    print("\nby category:")
    print(df.groupby("category").ceiling_f1_arabert
          .agg(["mean", "median", "min"]).round(4).to_string())
    print("\ntau candidates: p5 of ceiling = "
          f"{s.quantile(.05):.3f}; median-based margin (median-0.1) = "
          f"{s.median()-0.1:.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
