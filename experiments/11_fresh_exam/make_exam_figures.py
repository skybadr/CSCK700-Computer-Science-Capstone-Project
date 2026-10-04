"""Exp 11g — Figures for the fresh exam (same visual system as Exp 10)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT.parent / "10_final_benchmark"))
from make_figures import BLUE_SEQ, INK, INK2, MUTED, SERIES, plt, save  # noqa: E402


def fig_accuracy(ev, outdir):
    order = ["APCS-v2", "always llmlingua2@0.5", "APCS-1.0.0",
             "always llmlingua2@0.3", "random selection", "always llmlingua2@0.7",
             "always noop@1.0"]
    pretty = {"APCS-v2": "APCS-v2 (new)", "APCS-1.0.0": "APCS 1.0.0",
              "always noop@1.0": "never compress", "random selection": "random selection"}
    t = ev["policies"]
    acc = [t[n]["accuracy_best_balance"] for n in order]
    lo = [t[n]["accuracy_ci95"][0] for n in order]
    hi = [t[n]["accuracy_ci95"][1] for n in order]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.set_axisbelow(True)
    cols = [SERIES["llmlingua2"] if n == "APCS-v2" else
            BLUE_SEQ[3] if n == "APCS-1.0.0" else BLUE_SEQ[1] for n in order]
    ax.barh(y, acc, color=cols, height=0.62)
    ax.errorbar(acc, y, xerr=[np.array(acc) - lo, np.array(hi) - acc], fmt="none",
                ecolor=INK, capsize=2.5, lw=0.9)
    for yy, a, h in zip(y, acc, hi):
        ax.annotate(f"{a:.1%}", (h, yy), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [pretty.get(n, n.replace("always llmlingua2", "always LLMLingua-2"))
                      for n in order])
    ax.set_xlim(0, max(hi) * 1.18)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Prompts where the policy picks the best-balance option")
    ax.set_title(f"Fresh exam: {ev['n_prompts']} never-seen prompts\n"
                 "error bars: 95% bootstrap CI", fontsize=10.5)
    save(fig, outdir, "exam_fig1_selector_accuracy")


def fig_protected(sec, outdir):
    rates = ["@0.7", "@0.5", "@0.3"]
    plain = [sec["protected_vs_plain"][r]["llmlingua2"]["cost_change_pct"] for r in rates]
    prot = [sec["protected_vs_plain"][r]["llmlingua2_protected"]["cost_change_pct"]
            for r in rates]
    x = np.arange(len(rates))
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.set_axisbelow(True)
    w = 0.36
    ax.bar(x - w / 2, plain, w, color=SERIES["llmlingua2"], label="LLMLingua-2")
    ax.bar(x + w / 2, prot, w, color=SERIES["random_deletion"],
           label="LLMLingua-2, length instruction protected")
    for xx, v in zip(list(x - w / 2) + list(x + w / 2), plain + prot):
        ax.annotate(f"{v:+.1f}%", (xx, v), xytext=(0, 3 if v >= 0 else -11),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylim(min(prot + plain + [0]) - 8, max(plain) * 1.15)
    ax.set_xticks(x, [f"target rate {r[1:]}" for r in rates])
    ax.set_ylabel("Total API cost vs uncompressed prompt")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Protecting the length instruction removes the cost penalty\n"
                 f"fresh exam, {sec['n_with_length_instruction']} prompts that "
                 "contain one", fontsize=10.5)
    save(fig, outdir, "exam_fig2_protected_cost")


def main():
    outdir = RESULTS / "figures"
    outdir.mkdir(exist_ok=True)
    ev = json.loads((RESULTS / "exam_evaluation.json").read_text())
    sec = json.loads((RESULTS / "exam_secondary.json").read_text())
    fig_accuracy(ev, outdir)
    fig_protected(sec, outdir)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
