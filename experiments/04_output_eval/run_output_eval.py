"""Experiment 04 — Output-level evaluation (Phase 3).

Sends every unique prompt text from Experiment 03 (originals + compressed
variants) to gpt-4o-mini (temperature 0.0, max_tokens 512), then computes
output-level BERTScore F1 between the response to each compressed prompt and
the response to its original prompt. This is the decisive fidelity axis for
the method comparison and later Pareto labelling (Exp 02 Finding 4 / Exp 03
Findings 3-4: prompt-level F1 cannot rank methods).

Robustness: responses are checkpointed to responses.jsonl as they arrive;
rerunning resumes from the checkpoint (no double spend). Concurrency 8,
exponential-backoff retries.

Outputs (results/):
  responses.jsonl          one record per unique API call (checkpoint)
  output_eval_results.csv  Exp03 rows + response texts + output-level F1
  api_usage.json           actual token usage and cost
"""

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 512
CONCURRENCY = 8
PRICE_IN, PRICE_OUT = 0.15, 0.60  # USD per 1M tokens (checked 2026-07)

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent / "03_compression_benchmark" / "results" / "benchmark_results.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CKPT = RESULTS / "responses.jsonl"


def call_key(prompt_id: str, text: str) -> str:
    return hashlib.md5(f"{prompt_id}|{text}".encode("utf-8")).hexdigest()


def load_checkpoint() -> dict:
    done = {}
    if CKPT.exists():
        with CKPT.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["key"]] = rec
    return done


async def fetch_all(calls: list[dict]) -> None:
    from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
    client = AsyncOpenAI()  # key from OPENAI_API_KEY env
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    n_done, t0 = 0, time.time()

    async def one(call):
        nonlocal n_done
        async with sem:
            for attempt in range(6):
                try:
                    resp = await client.chat.completions.create(
                        model=MODEL, temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        messages=[{"role": "user", "content": call["text"]}])
                    rec = {"key": call["key"], "prompt_id": call["prompt_id"],
                           "response": resp.choices[0].message.content or "",
                           "prompt_tokens": resp.usage.prompt_tokens,
                           "completion_tokens": resp.usage.completion_tokens,
                           "finish_reason": resp.choices[0].finish_reason}
                    async with lock:
                        with CKPT.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_done += 1
                        if n_done % 100 == 0:
                            rate = n_done / (time.time() - t0)
                            print(f"  {n_done}/{len(calls)} responses "
                                  f"({rate:.1f}/s)", flush=True)
                    return
                except (RateLimitError, APIConnectionError, APIError) as e:
                    wait = min(2 ** attempt * 2, 60)
                    print(f"  retry {attempt+1} for {call['key'][:8]} "
                          f"({type(e).__name__}), waiting {wait}s", flush=True)
                    await asyncio.sleep(wait)
            raise RuntimeError(f"6 retries exhausted for {call['key']}")

    await asyncio.gather(*(one(c) for c in calls))


def main() -> None:
    df = pd.read_csv(BENCH)
    assert (df.error.fillna("") == "").all(), "Exp03 rows with errors present"

    # unique calls: one per distinct (prompt_id, text)
    uniq = df.drop_duplicates(subset=["prompt_id", "compressed"])[
        ["prompt_id", "compressed"]].rename(columns={"compressed": "text"})
    uniq["key"] = [call_key(r.prompt_id, r.text) for r in uniq.itertuples()]

    done = load_checkpoint()
    todo = [r._asdict() for r in uniq.itertuples(index=False)
            if r.key not in done]
    print(f"Unique calls: {len(uniq)} | already checkpointed: "
          f"{len(uniq) - len(todo)} | to fetch: {len(todo)}")

    if todo:
        asyncio.run(fetch_all(todo))
        done = load_checkpoint()

    # map responses back to all rows
    resp_map = {k: v["response"] for k, v in done.items()}
    df["call_key"] = [call_key(r.prompt_id, r.compressed)
                      for r in df.itertuples()]
    df["response"] = df.call_key.map(resp_map)
    missing = df.response.isna().sum()
    assert missing == 0, f"{missing} rows without responses"

    # reference response = the noop row's response for the same prompt
    ref_resp = df[df.method == "noop"].set_index("prompt_id").response
    df["ref_response"] = df.prompt_id.map(ref_resp)

    # output-level fidelity (skip noop: identical by construction)
    import torch
    from bert_score import score as bertscore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scored = df.method != "noop"
    for key, model, layers in [
            ("arabert", "aubmindlab/bert-base-arabertv02", 9),
            ("mbert", "bert-base-multilingual-cased", 9)]:
        print(f"Output-level BERTScore ({model}) on {int(scored.sum())} rows ...",
              flush=True)
        _, _, f1 = bertscore(df.loc[scored, "response"].tolist(),
                             df.loc[scored, "ref_response"].tolist(),
                             model_type=model, num_layers=layers,
                             batch_size=32, device=device)
        df.loc[scored, f"out_f1_{key}"] = f1.numpy().round(4)
        df.loc[~scored, f"out_f1_{key}"] = 1.0

    df.drop(columns=["ref_response"]).to_csv(
        RESULTS / "output_eval_results.csv", index=False, encoding="utf-8-sig")

    in_tok = sum(v["prompt_tokens"] for v in done.values())
    out_tok = sum(v["completion_tokens"] for v in done.values())
    truncated = sum(1 for v in done.values()
                    if v["finish_reason"] == "length")
    usage = {"model": MODEL, "temperature": TEMPERATURE,
             "max_tokens": MAX_TOKENS, "unique_calls": len(done),
             "prompt_tokens": in_tok, "completion_tokens": out_tok,
             "responses_truncated_at_512": truncated,
             "cost_usd": round(in_tok / 1e6 * PRICE_IN
                               + out_tok / 1e6 * PRICE_OUT, 4)}
    (RESULTS / "api_usage.json").write_text(json.dumps(usage, indent=2))
    print(json.dumps(usage, indent=2))

    print("\n=== Output-level F1 (AraBERT) by method x rate ===")
    print(df[scored].groupby(["method", "target_rate"])
          [["achieved_keep", "out_f1_arabert", "out_f1_mbert"]]
          .mean().round(3).to_string())
    print(f"\nSaved: {RESULTS / 'output_eval_results.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
