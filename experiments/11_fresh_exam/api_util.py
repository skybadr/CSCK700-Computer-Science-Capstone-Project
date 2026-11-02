"""Shared Luna call runner for Exp 11 (same model, parameters and pricing as
Exp 10: imports them from run_api.py so the protocol cannot drift)."""

import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "10_final_benchmark"))
import run_api  # noqa: E402

key_of = run_api.key_of
PRICE_IN, PRICE_OUT = run_api.PRICE_IN, run_api.PRICE_OUT


def load_responses(*paths):
    out = {}
    for p in paths:
        p = Path(p)
        if p.exists():
            for line in p.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    out[r["key"]] = r
    return out


async def _run(calls, ckpt, concurrency=8):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    params, snapshot = await run_api.resolve_params(client)
    print(f"model {run_api.MODEL} -> {snapshot} | {params}", flush=True)
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    st = {"n": 0, "cost": 0.0, "failed": 0, "abort": None, "t0": time.time()}

    async def one(c):
        async with sem:
            if st["abort"]:
                return
            for attempt in range(6):
                try:
                    r = await client.chat.completions.create(
                        model=run_api.MODEL,
                        messages=[{"role": "user", "content": c["text"]}], **params)
                    rec = {"key": c["key"], "prompt_id": c["prompt_id"],
                           "kind": c.get("kind", "main"), "model": r.model,
                           "response": r.choices[0].message.content or "",
                           "prompt_tokens": r.usage.prompt_tokens,
                           "completion_tokens": r.usage.completion_tokens,
                           "finish_reason": r.choices[0].finish_reason}
                    async with lock:
                        with open(ckpt, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        st["n"] += 1
                        st["cost"] += (r.usage.prompt_tokens * PRICE_IN
                                       + r.usage.completion_tokens * PRICE_OUT) / 1e6
                        if st["n"] % 500 == 0:
                            print(f"  {st['n']}/{len(calls)} (${st['cost']:.2f})",
                                  flush=True)
                    return
                except Exception as e:
                    if run_api.is_quota_error(e):
                        st["abort"] = "insufficient_quota"
                        return
                    await asyncio.sleep(min(2 ** attempt * 2, 60))
            st["failed"] += 1

    await asyncio.gather(*(one(c) for c in calls))
    print(f"done: {st['n']} new, {st['failed']} failed, ${st['cost']:.3f}"
          + (f" | ABORTED {st['abort']}" if st["abort"] else ""), flush=True)
    return st


def run_calls(calls, ckpt, seed=42):
    """calls: list of dicts with key, prompt_id, text (and optional kind).
    Already-checkpointed keys are skipped; order is shuffled with seed."""
    done = set(load_responses(ckpt))
    todo = [c for c in calls if c["key"] not in done]
    random.Random(seed).shuffle(todo)
    print(f"calls {len(calls)} | already done {len(calls) - len(todo)} | "
          f"todo {len(todo)}", flush=True)
    if not todo:
        return {"n": 0, "cost": 0.0, "failed": 0}
    return asyncio.run(_run(todo, ckpt))
