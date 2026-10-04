"""Exp 11d — The fresh exam run: compress, answer, score (400 new prompts).

Same protocol as Exp 10 (gpt-5.6-luna via api_util, AraBERT/mBERT scoring,
measured cost) plus the protected variant and QA correctness. tau is fixed
from Exp 10's dev ceiling, so no repeat calls are needed here.

Outputs: results/exam_features.csv, results/exam_local.csv,
         results/responses_exam.jsonl, results/exam_results.csv
"""

import json
import random
import sys
import traceback
from pathlib import Path

import pandas as pd
import tiktoken
import torch

from api_util import PRICE_IN, PRICE_OUT, key_of, load_responses, run_calls
from protect import compress_protected, has_length_instruction

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT.parent / "10_final_benchmark"))
from features_final import farasa_density  # noqa: E402
from qa_metrics import contains, recall  # noqa: E402

from apcs.features import extract_features  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")
RATES = [0.7, 0.5, 0.3]
SEED = 42
CATS = ["instruction", "summarisation", "qa", "creative"]


def random_deletion(text, rate, pid):
    rng = random.Random(f"{SEED}-{pid}-{rate}")
    w = text.split()
    keep = max(1, int(round(len(w) * rate)))
    return " ".join(w[i] for i in sorted(rng.sample(range(len(w)), keep)))


def main():
    exam = json.loads((PROJECT / "AraPromptBench_exam.json").read_text(encoding="utf-8"))["prompts"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- features (same definitions as the dev features)
    frows = []
    for p in exam:
        f = extract_features(p["prompt"]).as_dict()
        frows.append(dict(prompt_id=p["id"], task_category=p["category"], **f))
    feats = pd.DataFrame(frows).drop(columns=["morphological_density"])
    feats["morphological_density"] = farasa_density([p["prompt"] for p in exam])
    feats["has_length_instruction"] = [int(has_length_instruction(p["prompt"]))
                                       for p in exam]
    for c in CATS:
        feats[f"cat_{c}"] = (feats.task_category == c).astype(int)
    feats.to_csv(RESULTS / "exam_features.csv", index=False, encoding="utf-8-sig")

    # ---- compression
    from llmlingua import PromptCompressor
    qwen = PromptCompressor(model_name="Qwen/Qwen2.5-0.5B", device_map=device)
    ll2 = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True, device_map=device)
    rows = []
    for i, p in enumerate(exam):
        t, pid = p["prompt"], p["id"]
        ot = len(ENC.encode(t))
        base = dict(prompt_id=pid, category=p["category"], source=p["source"],
                    band=p["length_band"], orig_tokens=ot)
        rows.append(dict(**base, method="noop", target_rate=1.0, compressed=t,
                         error=""))
        for r in RATES:
            variants = {"random_deletion": lambda: random_deletion(t, r, pid),
                        "llmlingua_qwen": lambda: qwen.compress_prompt(t, rate=r)["compressed_prompt"],
                        "llmlingua2": lambda: ll2.compress_prompt(t, rate=r)["compressed_prompt"],
                        "llmlingua2_protected": lambda: compress_protected(ll2, t, r)}
            for m, fn in variants.items():
                try:
                    comp, err = fn(), ""
                except Exception:
                    comp, err = "", traceback.format_exc(limit=1).splitlines()[-1]
                rows.append(dict(**base, method=m, target_rate=r,
                                 compressed=comp, error=err))
        if (i + 1) % 100 == 0:
            print(f"  compressed {i + 1}/{len(exam)}", flush=True)
    del qwen, ll2
    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df = df[(df.error == "") & (df.compressed != "")].copy()
    df["comp_tokens"] = df.compressed.map(lambda s: len(ENC.encode(s)))
    df["achieved_keep"] = (df.comp_tokens / df.orig_tokens).round(4)
    df["tcr"] = (1 - df.achieved_keep).round(4)
    df.to_csv(RESULTS / "exam_local.csv", index=False, encoding="utf-8-sig")

    # ---- answers (one call per unique text)
    ckpt = RESULTS / "responses_exam.jsonl"
    df["call_key"] = [key_of(p, t) for p, t in zip(df.prompt_id, df.compressed)]
    calls = list({k: {"key": k, "prompt_id": p, "text": t} for k, p, t in
                  zip(df.call_key, df.prompt_id, df.compressed)}.values())
    run_calls(calls, ckpt)
    got = load_responses(ckpt)
    g = lambda k, f: got.get(k, {}).get(f)
    df["response"] = [g(k, "response") for k in df.call_key]
    df["resp_prompt_tokens"] = [g(k, "prompt_tokens") for k in df.call_key]
    df["resp_completion_tokens"] = [g(k, "completion_tokens") for k in df.call_key]
    df["resp_truncated"] = [g(k, "finish_reason") == "length" for k in df.call_key]
    ref = df[df.method == "noop"].set_index("prompt_id").response.dropna()
    df = df[df.prompt_id.isin(ref.index) & df.response.notna()].copy()

    # ---- scoring
    from score_and_analyze import bertscore_f1
    scored = df.method != "noop"
    for s in ["arabert", "mbert"]:
        df.loc[scored, f"out_f1_{s}"] = bertscore_f1(
            df.loc[scored, "response"].tolist(),
            ref.loc[df.loc[scored, "prompt_id"]].tolist(), s)
        df.loc[~scored, f"out_f1_{s}"] = 1.0
    df["cost_usd"] = (df.resp_prompt_tokens * PRICE_IN
                      + df.resp_completion_tokens * PRICE_OUT) / 1e6
    gold = {p["id"]: p["gold_answers"] for p in exam if p.get("gold_answers")}
    df["qa_contains"] = [contains(a, gold[p]) if p in gold else None
                         for a, p in zip(df.response, df.prompt_id)]
    df["qa_recall"] = [recall(a, gold[p]) if p in gold else None
                       for a, p in zip(df.response, df.prompt_id)]
    df.to_csv(RESULTS / "exam_results.csv", index=False, encoding="utf-8-sig")
    print(f"exam_results.csv: {len(df)} rows, {df.prompt_id.nunique()} prompts")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
