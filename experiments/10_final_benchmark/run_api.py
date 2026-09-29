"""Exp 10b — API measurement: responses for every unique prompt text + the
repeat-call ceiling, on the LLM under test.

LLM under test: gpt-5.6-luna (decision 2026-09-29: newest-generation
budget-tier model; replaces gpt-5.4-mini, whose partial July run is archived
in results/archive_gpt54mini/ and not used for any reported result).
The exact snapshot string returned by the API is stored on every response so
model drift within a run is detectable.

Parameter resolution: probes from the most deterministic / cheapest
configuration down, records the first one the model accepts.

Safety:
  - every response checkpointed to responses.jsonl (rerun = resume)
  - calls issued in a seeded random order, so partial progress is a random
    sample and the early cost projection is representative
  - early projection check after EARLY_CHECK_N calls: abort if the projected
    total exceeds EARLY_ABORT_USD
  - hard budget guard: abort if actual spend exceeds BUDGET_USD
  - insufficient_quota aborts immediately (no retry storm)

Usage:
  python run_api.py --smoke      # 6 test calls, prints responses + cost projection
  python run_api.py              # full run (needs OPENAI_API_KEY)
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CKPT = RESULTS / "responses.jsonl"

MODEL = "gpt-5.6-luna"
PRICE_IN, PRICE_OUT = 0.20, 1.20   # USD per 1M tokens, verified 2026-09-29
MAX_OUT = 1024
CONCURRENCY = 8
BUDGET_USD = 9.0
EARLY_CHECK_N = 500
EARLY_ABORT_USD = 6.0
SEED = 42

PARAM_CANDIDATES = [
    {"reasoning_effort": "none", "temperature": 0.0,
     "max_completion_tokens": MAX_OUT},
    {"reasoning_effort": "minimal", "max_completion_tokens": MAX_OUT},
    {"reasoning_effort": "low", "max_completion_tokens": MAX_OUT},
    {"max_completion_tokens": MAX_OUT},
]


def key_of(pid, text, repeat=False):
    h = hashlib.md5(f"{pid}|{text}".encode("utf-8")).hexdigest()
    return f"{h}-r2" if repeat else h


def is_quota_error(e):
    return "insufficient_quota" in str(e)


async def resolve_params(client):
    errors = []
    for params in PARAM_CANDIDATES:
        try:
            r = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": "قل نعم"}],
                **params)
            return params, r.model
        except Exception as e:
            if is_quota_error(e):
                raise SystemExit("OpenAI account has insufficient quota — "
                                 "top up billing, then rerun.")
            errors.append(f"{params}: {str(e)[:120]}")
    raise SystemExit("no parameter set accepted:\n" + "\n".join(errors))


def build_calls():
    df = pd.read_csv(RESULTS / "local_results.csv")
    df = df[(df.error.fillna("") == "") & (df.compressed.fillna("") != "")]
    calls = {}
    for r in df.itertuples():
        k = key_of(r.prompt_id, r.compressed)
        calls.setdefault(k, {"key": k, "prompt_id": r.prompt_id,
                             "text": r.compressed, "kind": "main"})
    for r in df[df.method == "noop"].itertuples():
        k = key_of(r.prompt_id, r.compressed, repeat=True)
        calls[k] = {"key": k, "prompt_id": r.prompt_id,
                    "text": r.compressed, "kind": "ceiling"}
    return df, calls


async def smoke(client, params, snapshot):
    """Six real prompts (one per category + two long) -> cost projection."""
    df, calls = build_calls()
    noop = df[df.method == "noop"]
    picks = []
    for cat in ["instruction", "summarisation", "qa", "creative"]:
        picks.append(noop[noop.category == cat].iloc[0])
    picks += list(noop.sort_values("orig_tokens").iloc[-2:].itertuples(index=False))
    usage = []
    for p in picks:
        text = p["compressed"] if isinstance(p, pd.Series) else p.compressed
        r = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": text}], **params)
        out = r.choices[0].message.content or ""
        usage.append((r.usage.prompt_tokens, r.usage.completion_tokens,
                      r.choices[0].finish_reason))
        print(f"--- {len(text)} chars in | {r.usage.prompt_tokens} in / "
              f"{r.usage.completion_tokens} out tok | {r.choices[0].finish_reason}")
        print(out[:300].replace("\n", " "))
    mean_in = sum(u[0] for u in usage) / len(usage)
    mean_out = sum(u[1] for u in usage) / len(usage)
    per_call = (mean_in * PRICE_IN + mean_out * PRICE_OUT) / 1e6
    print(f"\nsnapshot: {snapshot} | params: {params}")
    print(f"mean in {mean_in:.0f} / out {mean_out:.0f} tokens per call")
    print(f"projected full run: {len(calls)} calls x ${per_call:.6f} "
          f"= ${len(calls) * per_call:.2f} (sample of 6 — skewed long; "
          f"the early check at {EARLY_CHECK_N} calls is the reliable estimate)")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    params, snapshot = await resolve_params(client)
    print(f"model {MODEL} -> snapshot {snapshot} | params {params} | "
          f"${PRICE_IN}/{PRICE_OUT} per M")
    if args.smoke:
        await smoke(client, params, snapshot)
        return

    (RESULTS / "api_config.json").write_text(json.dumps(
        {"model": MODEL, "snapshot_at_start": snapshot, "params": params,
         "price_per_M": {"in": PRICE_IN, "out": PRICE_OUT},
         "budget_usd": BUDGET_USD,
         "resolved_utc": datetime.now(timezone.utc).isoformat()}, indent=2))

    _, calls = build_calls()
    done = set()
    if CKPT.exists():
        with CKPT.open(encoding="utf-8") as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    todo = [c for k, c in calls.items() if k not in done]
    random.Random(SEED).shuffle(todo)
    total = len(calls)
    print(f"unique calls {total} | done {total - len(todo)} | todo {len(todo)}")

    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    state = {"n": 0, "cost": 0.0, "t0": time.time(), "abort": None,
             "failed": 0}

    async def one(c):
        async with sem:
            if state["abort"]:
                return
            for attempt in range(6):
                try:
                    r = await client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": c["text"]}],
                        **params)
                    rec = {"key": c["key"], "prompt_id": c["prompt_id"],
                           "kind": c["kind"], "model": r.model,
                           "response": r.choices[0].message.content or "",
                           "prompt_tokens": r.usage.prompt_tokens,
                           "completion_tokens": r.usage.completion_tokens,
                           "finish_reason": r.choices[0].finish_reason}
                    async with lock:
                        with CKPT.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        state["n"] += 1
                        state["cost"] += (r.usage.prompt_tokens * PRICE_IN
                                          + r.usage.completion_tokens
                                          * PRICE_OUT) / 1e6
                        n, cost = state["n"], state["cost"]
                        if n == EARLY_CHECK_N:
                            proj = cost / n * len(todo)
                            print(f"EARLY CHECK: ${cost:.3f} for {n} calls -> "
                                  f"projected ${proj:.2f} for this run",
                                  flush=True)
                            if proj > EARLY_ABORT_USD:
                                state["abort"] = (f"projected ${proj:.2f} > "
                                                  f"${EARLY_ABORT_USD}")
                        if cost > BUDGET_USD:
                            state["abort"] = f"budget guard ${cost:.2f}"
                        if n % 1000 == 0:
                            rate = n / (time.time() - state["t0"])
                            print(f"  {n}/{len(todo)} ({rate:.1f}/s, "
                                  f"${cost:.2f})", flush=True)
                    return
                except Exception as e:
                    if is_quota_error(e):
                        state["abort"] = "insufficient_quota"
                        return
                    await asyncio.sleep(min(2 ** attempt * 2, 60))
            state["failed"] += 1
            print(f"  FAILED after retries: {c['key'][:8]}", flush=True)

    await asyncio.gather(*(one(c) for c in todo))
    msg = (f"done: {state['n']} new responses, {state['failed']} failed, "
           f"session cost ${state['cost']:.2f}")
    if state["abort"]:
        msg += f" | ABORTED: {state['abort']}"
    print(msg, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
