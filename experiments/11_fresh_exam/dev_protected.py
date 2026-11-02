"""Exp 11b — Measure the length-instruction-protected LLMLingua-2 variant on
the v2 DEV prompts, so the cost-aware selector can learn when it pays off.

Prompts without a length instruction compress exactly like plain
LLMLingua-2, so they reuse the Exp 10 text and answer. Only changed texts
are sent to Luna (same protocol, via api_util). Output rows have the same
columns as Exp 10's final_results.csv.

Output: results/dev_protected.csv, results/responses_dev_protected.jsonl
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
import torch

from api_util import key_of, load_responses, run_calls, PRICE_IN, PRICE_OUT
from protect import compress_protected, has_length_instruction

ROOT = Path(__file__).resolve().parent
EXP10 = ROOT.parent / "10_final_benchmark" / "results"
RESULTS = ROOT / "results"
ENC = tiktoken.get_encoding("cl100k_base")
RATES = [0.7, 0.5, 0.3]
METHOD = "llmlingua2_protected"


def main():
    RESULTS.mkdir(exist_ok=True)
    fr = pd.read_csv(EXP10 / "final_results.csv")
    dev = fr[(fr.origin == "v2") & (fr.split == "dev")]
    noop = dev[dev.method == "noop"].set_index("prompt_id")
    plain = dev[dev.method == "llmlingua2"].set_index(["prompt_id", "target_rate"])

    from llmlingua import PromptCompressor
    pc = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True, device_map="cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for pid, r in noop.iterrows():
        text = r.compressed
        flag = has_length_instruction(text)
        for rate in RATES:
            comp = (compress_protected(pc, text, rate) if flag
                    else plain.loc[(pid, rate), "compressed"])
            ct = len(ENC.encode(comp))
            rows.append(dict(prompt_id=pid, category=r.category, origin="v2",
                             split="dev", source=r.source, band=r.band,
                             method=METHOD, target_rate=rate, compressed=comp,
                             orig_tokens=r.orig_tokens, comp_tokens=ct,
                             achieved_keep=round(ct / r.orig_tokens, 4),
                             tcr=round(1 - ct / r.orig_tokens, 4),
                             has_length_instruction=flag))
    df = pd.DataFrame(rows)
    print(f"protected variant: {int(df.has_length_instruction.sum() / 3)} of "
          f"{len(noop)} dev prompts carry a length instruction", flush=True)

    ckpt = RESULTS / "responses_dev_protected.jsonl"
    known = load_responses(EXP10 / "responses.jsonl", ckpt)
    df["call_key"] = [key_of(p, t) for p, t in zip(df.prompt_id, df.compressed)]
    calls = [{"key": k, "prompt_id": p, "text": t}
             for k, p, t in zip(df.call_key, df.prompt_id, df.compressed)
             if k not in known]
    calls = list({c["key"]: c for c in calls}.values())
    run_calls(calls, ckpt)
    known = load_responses(EXP10 / "responses.jsonl", ckpt)

    get = lambda k, f: known.get(k, {}).get(f)
    df["response"] = [get(k, "response") for k in df.call_key]
    df["resp_prompt_tokens"] = [get(k, "prompt_tokens") for k in df.call_key]
    df["resp_completion_tokens"] = [get(k, "completion_tokens") for k in df.call_key]
    df = df[df.response.notna()].copy()
    ref = noop.response

    sys.path.insert(0, str(ROOT.parent / "10_final_benchmark"))
    from score_and_analyze import bertscore_f1
    for s in ["arabert", "mbert"]:
        df[f"out_f1_{s}"] = bertscore_f1(df.response.tolist(),
                                         ref.loc[df.prompt_id].tolist(), s)
    df["cost_usd"] = (df.resp_prompt_tokens * PRICE_IN
                      + df.resp_completion_tokens * PRICE_OUT) / 1e6
    df.to_csv(RESULTS / "dev_protected.csv", index=False, encoding="utf-8-sig")

    # quick comparison vs plain LLMLingua-2 on the prompts that differ
    plain_df = dev[dev.method == "llmlingua2"].copy()
    plain_df["cost_usd"] = (plain_df.resp_prompt_tokens * PRICE_IN
                            + plain_df.resp_completion_tokens * PRICE_OUT) / 1e6
    base_cost = (noop.resp_prompt_tokens * PRICE_IN
                 + noop.resp_completion_tokens * PRICE_OUT) / 1e6
    flagged = df[df.has_length_instruction].prompt_id.unique()
    print("\nprompts WITH a length instruction (where the variant differs):")
    for rate in RATES:
        p = df[(df.target_rate == rate) & df.prompt_id.isin(flagged)].set_index("prompt_id")
        q = plain_df[(plain_df.target_rate == rate)
                     & plain_df.prompt_id.isin(flagged)].set_index("prompt_id")
        b = base_cost.loc[p.index]
        print(f"  rate {rate}: TCR plain {q.tcr.mean():.3f} vs protected "
              f"{p.tcr.mean():.3f} | cost vs uncompressed: plain "
              f"{100 * (q.cost_usd.sum() / b.sum() - 1):+.1f}% vs protected "
              f"{100 * (p.cost_usd.sum() / b.sum() - 1):+.1f}% | out-F1 plain "
              f"{q.out_f1_arabert.mean():.3f} vs protected {p.out_f1_arabert.mean():.3f}")
    print("saved dev_protected.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
