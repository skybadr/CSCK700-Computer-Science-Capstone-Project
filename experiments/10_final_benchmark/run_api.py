"""Exp 10b — API measurement: responses for every unique prompt text + the
repeat-call ceiling, on the current small-tier OpenAI model.

Model resolution: prefers gpt-5.4-mini (protocol decision after gpt-4o-mini's
deprecation), falls back down the candidate list to whatever the account can
actually call, and records the resolved model + parameters in the manifest.
Reasoning models: reasoning_effort='minimal' (cost + determinism-proximity),
max_completion_tokens instead of max_tokens, temperature omitted if rejected.

Safety: every response checkpointed to responses.jsonl (rerun = resume);
hard budget guard aborts the loop if projected spend exceeds BUDGET_USD.

Usage: python run_api.py            (needs OPENAI_API_KEY)
"""

import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CKPT = RESULTS / "responses.jsonl"

CANDIDATES = ["gpt-5.4-mini", "gpt-5-mini", "gpt-4o-mini"]
PRICES = {  # USD per 1M tokens (in, out) — verified 2026-07
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4o-mini": (0.15, 0.60),
}
MAX_OUT = 1024
CONCURRENCY = 8
BUDGET_USD = 60.0


def key_of(pid, text, repeat=False):
    h = hashlib.md5(f"{pid}|{text}".encode("utf-8")).hexdigest()
    return f"{h}-r2" if repeat else h


async def resolve_model(client):
    avail = {m.id for m in (await client.models.list()).data}
    model = next((c for c in CANDIDATES if c in avail), None)
    if model is None:
        # try prefix match (dated snapshots)
        for c in CANDIDATES:
            hit = sorted(m for m in avail if m.startswith(c))
            if hit:
                model = hit[0]
                break
    if model is None:
        raise SystemExit(f"none of {CANDIDATES} available; models: "
                         f"{sorted(list(avail))[:40]}")
    # parameter probe
    for params in (
        {"reasoning_effort": "minimal", "max_completion_tokens": MAX_OUT},
        {"max_completion_tokens": MAX_OUT},
        {"max_tokens": 512, "temperature": 0.0},
    ):
        try:
            await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "قل نعم"}],
                **params)
            return model, params
        except Exception as e:
            last = e
    raise SystemExit(f"no parameter combination accepted by {model}: {last}")


async def main():
    RESULTS.mkdir(exist_ok=True)
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
    done = set()
    if CKPT.exists():
        with CKPT.open(encoding="utf-8") as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    todo = [c for k, c in calls.items() if k not in done]
    print(f"unique calls {len(calls)} | done {len(calls)-len(todo)} | "
          f"todo {len(todo)}")

    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    model, params = await resolve_model(client)
    pin, pout = PRICES.get(model.split(":")[0],
                           PRICES.get(next((c for c in CANDIDATES
                                            if model.startswith(c)),
                                           "gpt-4o-mini")))
    print(f"model: {model} | params: {params} | ${pin}/{pout} per M")
    (RESULTS / "api_config.json").write_text(json.dumps(
        {"model": model, "params": params,
         "resolved_utc": datetime.now(timezone.utc).isoformat()}, indent=2))

    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    state = {"n": 0, "cost": 0.0, "t0": time.time(), "abort": False}

    async def one(c):
        async with sem:
            if state["abort"]:
                return
            for attempt in range(6):
                try:
                    r = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": c["text"]}],
                        **params)
                    rec = {"key": c["key"], "prompt_id": c["prompt_id"],
                           "kind": c["kind"],
                           "response": r.choices[0].message.content or "",
                           "prompt_tokens": r.usage.prompt_tokens,
                           "completion_tokens": r.usage.completion_tokens,
                           "finish_reason": r.choices[0].finish_reason}
                    async with lock:
                        with CKPT.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        state["n"] += 1
                        state["cost"] += (r.usage.prompt_tokens * pin
                                          + r.usage.completion_tokens * pout) / 1e6
                        if state["cost"] > BUDGET_USD:
                            state["abort"] = True
                            print(f"BUDGET GUARD tripped at ${state['cost']:.2f}",
                                  flush=True)
                        if state["n"] % 500 == 0:
                            rate = state["n"] / (time.time() - state["t0"])
                            print(f"  {state['n']}/{len(todo)} "
                                  f"({rate:.1f}/s, ${state['cost']:.2f})",
                                  flush=True)
                    return
                except Exception:
                    await asyncio.sleep(min(2 ** attempt * 2, 60))
            print(f"  FAILED after retries: {c['key'][:8]}", flush=True)

    await asyncio.gather(*(one(c) for c in todo))
    print(f"done: {state['n']} new responses, session cost ${state['cost']:.2f}"
          f"{' (ABORTED ON BUDGET)' if state['abort'] else ''}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
