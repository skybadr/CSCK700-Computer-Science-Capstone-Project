"""Exp 07c — How much output-level scoring is affected by AraBERT's 512 window?

BERTScore truncates inputs to the scorer's max sequence length. gpt-4o-mini
responses run up to 512 cl100k tokens, which can exceed AraBERT's 512-wordpiece
window. Quantifies: % of Exp 04 response pairs where either side exceeds the
window, and how out-F1 differs for affected vs unaffected pairs.
"""

import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
EXP04 = ROOT.parent / "04_output_eval" / "results" / "output_eval_results.csv"

WINDOW = 510  # 512 minus [CLS]/[SEP]


def main() -> None:
    df = pd.read_csv(EXP04)
    tok = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")

    # wordpiece length of every unique response (cache by text hash)
    uniq = df.response.drop_duplicates()
    print(f"tokenising {len(uniq)} unique responses with AraBERT tokenizer ...")
    lengths = {t: len(tok(t, add_special_tokens=False)["input_ids"])
               for t in uniq}
    df["resp_wp"] = df.response.map(lengths)
    ref = df[df.method == "noop"].set_index("prompt_id").resp_wp
    df["ref_wp"] = df.prompt_id.map(ref)
    df["window_exceeded"] = (df.resp_wp > WINDOW) | (df.ref_wp > WINDOW)

    sc = df[df.method != "noop"].copy()
    out = sc.groupby(["category", "window_exceeded"]).agg(
        n=("prompt_id", "size"),
        mean_out_f1=("out_f1_arabert", "mean")).round(3)
    sc[["prompt_id", "category", "method", "target_rate", "resp_wp",
        "ref_wp", "window_exceeded", "out_f1_arabert"]].to_csv(
        RESULTS / "scorer_window.csv", index=False, encoding="utf-8-sig")

    pct = 100 * sc.window_exceeded.mean()
    print(f"\npairs where either response exceeds {WINDOW} AraBERT wordpieces: "
          f"{pct:.1f}%")
    print("\nby category:")
    print((100 * sc.groupby('category').window_exceeded.mean()).round(1)
          .to_string())
    print("\nmean out-F1, affected vs not:")
    print(out.to_string())
    print("\nwordpiece length distribution of responses (all): "
          f"median {df.resp_wp.median():.0f}, p90 "
          f"{df.resp_wp.quantile(.9):.0f}, max {df.resp_wp.max():.0f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
