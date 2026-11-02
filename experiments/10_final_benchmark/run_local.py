"""Exp 10a — Final local compression benchmark: v2 (1,000) + probe pool (456).

Methods per pilot decisions: noop, random_deletion, llmlingua_qwen
(Qwen2.5-0.5B), llmlingua2 (XLM-R) at target rates 0.7/0.5/0.3.
Prompt-level BERTScore (AraBERT primary, mBERT sensitivity) computed locally.
Probe prompts are measured in the SAME run as v2 so origin is never
confounded with measurement conditions (Exp 09 requirement).

Output: results/local_results.csv + manifest.json
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
PROJECT = ROOT.parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
ENC = tiktoken.get_encoding("cl100k_base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ntok(t):
    return len(ENC.encode(t))


def random_deletion(text, rate, pid):
    rng = random.Random(f"{SEED}-{pid}-{rate}")
    words = text.split()
    keep = max(1, int(round(len(words) * rate)))
    idx = sorted(rng.sample(range(len(words)), keep))
    return " ".join(words[i] for i in idx)


def main():
    t0 = time.time()
    v2 = json.loads((PROJECT / "AraPromptBench_v2.json").read_text(encoding="utf-8"))
    probe = json.loads((ROOT.parent / "09_synthetic_sensitivity" / "probe_pool.json")
                       .read_text(encoding="utf-8"))
    prompts = [dict(id=p["id"], text=p["prompt"], category=p["category"],
                    origin="v2", split=p["split"], source=p["source"],
                    band=p["length_band"]) for p in v2["prompts"]]
    prompts += [dict(id=p["id"], text=p["prompt"], category=p["category"],
                     origin="probe", split="probe", source=p["source"],
                     band=p["length_band"]) for p in probe["prompts"]]
    print(f"{len(prompts)} prompts ({sum(p['origin']=='v2' for p in prompts)} v2 "
          f"+ {sum(p['origin']=='probe' for p in prompts)} probe) | {DEVICE}")

    from llmlingua import PromptCompressor
    compressors = {
        "llmlingua_qwen": PromptCompressor(model_name="Qwen/Qwen2.5-0.5B",
                                           device_map=DEVICE),
        "llmlingua2": PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True, device_map=DEVICE),
    }

    rows = []
    for i, p in enumerate(prompts):
        text, pid = p["text"], p["id"]
        base = dict(prompt_id=pid, category=p["category"], origin=p["origin"],
                    split=p["split"], source=p["source"], band=p["band"])
        ot = ntok(text)
        rows.append(dict(**base, method="noop", target_rate=1.0,
                         compressed=text, orig_tokens=ot, comp_tokens=ot,
                         error=""))
        for rate in RATES:
            comp = random_deletion(text, rate, pid)
            rows.append(dict(**base, method="random_deletion",
                             target_rate=rate, compressed=comp,
                             orig_tokens=ot, comp_tokens=ntok(comp), error=""))
            for name, pc in compressors.items():
                try:
                    out = pc.compress_prompt(text, rate=rate)
                    comp, err = out["compressed_prompt"], ""
                except Exception:
                    comp, err = "", traceback.format_exc(limit=1).splitlines()[-1]
                rows.append(dict(**base, method=name, target_rate=rate,
                                 compressed=comp, orig_tokens=ot,
                                 comp_tokens=ntok(comp) if comp else 0,
                                 error=err))
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(prompts)}] {time.time()-t0:.0f}s", flush=True)

    del compressors
    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df["achieved_keep"] = (df.comp_tokens / df.orig_tokens).round(4)
    df["tcr"] = (1 - df.achieved_keep).round(4)

    from bert_score import score as bertscore
    refs = df.prompt_id.map({p["id"]: p["text"] for p in prompts})
    ok = (df.error == "") & (df.compressed != "")
    for key, model in [("arabert", "aubmindlab/bert-base-arabertv02"),
                       ("mbert", "bert-base-multilingual-cased")]:
        print(f"prompt-level BERTScore ({key}) on {int(ok.sum())} rows ...",
              flush=True)
        _, _, f1 = bertscore(df.loc[ok, "compressed"].tolist(),
                             refs[ok].tolist(), model_type=model,
                             num_layers=9, batch_size=64, device=DEVICE)
        df.loc[ok, f"f1_{key}"] = f1.numpy().round(4)

    df.to_csv(RESULTS / "local_results.csv", index=False, encoding="utf-8-sig")
    import bert_score as bs
    import llmlingua as ll
    import transformers
    manifest = dict(
        experiment="10_final_benchmark_local", seed=SEED, rates=RATES,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        n_prompts=len(prompts),
        dataset_md5=dict(
            v2=hashlib.md5((PROJECT / "AraPromptBench_v2.json").read_bytes()).hexdigest(),
            probe=hashlib.md5((ROOT.parent / "09_synthetic_sensitivity" /
                               "probe_pool.json").read_bytes()).hexdigest()),
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      transformers=transformers.__version__,
                      llmlingua=getattr(ll, "__version__", "?"),
                      bert_score=bs.__version__),
        device=DEVICE + (f" ({torch.cuda.get_device_name(0)})"
                         if DEVICE == "cuda" else ""),
        runtime_s=round(time.time() - t0, 1))
    (RESULTS / "manifest_local.json").write_text(json.dumps(manifest, indent=2))
    print(df[ok].groupby(["method", "target_rate"])
          [["achieved_keep", "f1_arabert"]].mean().round(3).to_string())
    print(f"errors: {int((~ok).sum())} | runtime {time.time()-t0:.0f}s")
    print("saved: local_results.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
