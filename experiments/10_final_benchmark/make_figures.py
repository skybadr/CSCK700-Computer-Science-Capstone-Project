"""Exp 10f — Thesis-ready figures (PNG 300 dpi + PDF) and tables (Markdown).

Reads whatever outputs exist in --indir and skips figures whose inputs are
missing, so it can run on partial or dry-run outputs. Print-oriented light
theme; categorical colours use the validated reference palette in fixed slot
order (blue, green, magenta, yellow) and every series is also direct-labelled
so identity never relies on colour alone.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SYN = ROOT.parent / "09_synthetic_sensitivity" / "results"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SERIES = {"llmlingua2": "#2a78d6", "random_deletion": "#008300",
          "llmlingua_qwen": "#e87ba4", "noop": "#eda100"}
LABEL = {"llmlingua2": "LLMLingua-2", "random_deletion": "Random deletion",
         "llmlingua_qwen": "LLMLingua (Qwen scorer)", "noop": "No compression"}
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
DIVERGING = ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a89", "#c93a3a"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False,
})


def save(fig, outdir, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}")


def fig_tradeoff(res, tau, outdir):
    dev = res[(res.origin == "v2") & (res.split == "dev")]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in ["llmlingua2", "random_deletion", "llmlingua_qwen"]:
        g = dev[dev.method == m].groupby("target_rate")[
            ["tcr", "out_f1_arabert"]].mean().sort_index(ascending=False)
        ax.plot(g.tcr, g.out_f1_arabert, "-o", color=SERIES[m], lw=2, ms=6,
                label=LABEL[m])
        last = g.iloc[-1]
        ax.annotate(LABEL[m], (last.tcr, last.out_f1_arabert),
                    xytext=(6, -3), textcoords="offset points",
                    color=INK2, fontsize=9)
    ax.plot([0], [1.0], "o", color=SERIES["noop"], ms=7)
    ax.annotate(LABEL["noop"], (0, 1.0), xytext=(6, -10),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.axhline(tau, color=MUTED, ls="--", lw=1)
    ax.annotate(f"fidelity threshold τ = {tau}", (0.01, tau),
                xytext=(0, 4), textcoords="offset points", color=MUTED,
                fontsize=8.5)
    ax.set_xlabel("Token compression ratio (share of tokens removed)")
    ax.set_ylabel("Output-level BERTScore-F1 (AraBERT)")
    ax.set_title("Compression trade-off by method (dev set, points = target rates)")
    ax.legend(loc="lower left", fontsize=8.5)
    save(fig, outdir, "fig1_method_tradeoff")


def fig_flip(an, outdir):
    p = an["rq1_prompt_level"].get("llmlingua2_vs_random@0.5", {})
    o = an["rq1_output_level"].get("llmlingua2_vs_random@0.5", {})
    if "a_mean" not in p or "a_mean" not in o:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for (m, key), col in [(("llmlingua2", "a_mean"), SERIES["llmlingua2"]),
                          (("random_deletion", "b_mean"),
                           SERIES["random_deletion"])]:
        ys = [p[key], o[key]]
        ax.plot([0, 1], ys, "-o", color=col, lw=2.2, ms=7)
        ax.annotate(f"{LABEL[m]}  {ys[0]:.3f}", (0, ys[0]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center",
                    color=INK2, fontsize=9)
        ax.annotate(f"{ys[1]:.3f}", (1, ys[1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK2,
                    fontsize=9)
    ax.set_xticks([0, 1], ["Prompt-level F1\n(prompt vs prompt)",
                           "Output-level F1\n(answer vs answer)"])
    ax.set_xlim(-1.0, 1.35)
    ax.set_ylabel("Mean BERTScore-F1 (AraBERT)")
    ax.set_title("The ranking flips when you judge the answers (rate 0.5)")
    save(fig, outdir, "fig2_ranking_flip")


def fig_ceiling(ceil, tau, outdir):
    d = ceil[(ceil.origin == "v2") & (ceil.split == "dev")].ceiling_f1_arabert
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.hist(d, bins=30, color=BLUE_SEQ[3], edgecolor=SURFACE, lw=0.8)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.18)
    for x, lab, col in [(tau, f"adopted τ = {tau}", INK),
                        (0.85, "SDR threshold 0.85", MUTED)]:
        ax.axvline(x, color=col, ls="--", lw=1.2)
        ax.annotate(lab, (x, top * 1.08), xytext=(4, 0),
                    textcoords="offset points", color=col, fontsize=8.5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.set_xlabel("BERTScore-F1 between two answers to the IDENTICAL prompt")
    ax.set_ylabel("Prompts (dev)")
    ax.set_title("LLM noise ceiling\nidentical prompts do not give identical "
                 "answers", fontsize=10.5)
    save(fig, outdir, "fig3_noise_ceiling")


def fig_apcs(ap, outdir):
    p = ap["primary_arabert"]
    order = ["APCS", "always llmlingua2@0.7", "always noop@1.0",
             "always llmlingua2@0.5", "always llmlingua_qwen@0.5",
             "random selection", "always llmlingua2@0.3"]
    names = [n for n in order if n in p["test_policies"]]
    # always-LLMLingua can only be correct under the extended label set, so
    # it is plotted on that basis (marked *), matching the paired comparison
    ext = ["llmlingua_qwen" in n for n in names]
    accs = [p["test_policies"][n]["accuracy_extended" if e else "accuracy"]
            for n, e in zip(names, ext)]
    lo, hi = p["apcs_accuracy_ci95"]
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.set_axisbelow(True)
    y = np.arange(len(names))[::-1]
    cols = [SERIES["llmlingua2"] if n == "APCS" else BLUE_SEQ[1] for n in names]
    ax.barh(y, accs, color=cols, height=0.62)
    i = names.index("APCS")
    ax.errorbar(accs[i], y[i], xerr=[[accs[i] - lo], [hi - accs[i]]],
                color=INK, capsize=3, lw=1)
    pretty = {"always noop@1.0": "never compress",
              "random selection": "random selection"}
    ax.set_yticks(y, [pretty.get(n, n.replace("llmlingua_qwen", "LLMLingua")
                                 .replace("llmlingua2", "LLMLingua-2"))
                      + (" *" if e else "") for n, e in zip(names, ext)])
    for yy, a, n in zip(y, accs, names):
        x = hi if n == "APCS" else a
        ax.annotate(f"{a:.1%}", (x, yy), xytext=(5, 0),
                    textcoords="offset points", va="center", color=INK,
                    fontsize=9, fontweight="bold" if n == "APCS" else None)
    ax.set_xlabel("Test prompts where the policy picks the best-balance option")
    ax.set_xlim(0, max(max(accs), hi) * 1.22)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"APCS vs fixed strategies, held-out test set "
                 f"(n = {ap['n_test']})\nerror bar: 95% bootstrap CI on APCS",
                 fontsize=10.5)
    fig.text(0.01, 0.01, "* scored on extended labels that admit LLMLingua "
             "options (under primary labels it cannot be correct)",
             fontsize=7.5, color=MUTED)
    save(fig, outdir, "fig4_apcs_vs_baselines")


def fig_confusion(ap, outdir):
    cm = pd.DataFrame(ap["primary_arabert"]["confusion_matrix"])
    order = ["noop@1.0", "llmlingua2@0.7", "llmlingua2@0.5", "llmlingua2@0.3"]
    cm = cm.reindex(index=order, columns=order).fillna(0)
    short = ["none", "LL2 0.7", "LL2 0.5", "LL2 0.3"]
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(cm.values, cmap=matplotlib.colors.ListedColormap(BLUE_SEQ))
    for (i, j), v in np.ndenumerate(cm.values):
        ax.text(j, i, int(v), ha="center", va="center",
                color=SURFACE if v > cm.values.max() * 0.55 else INK)
    ax.set_xticks(range(4), short)
    ax.set_yticks(range(4), short)
    ax.set_xlabel("APCS recommendation")
    ax.set_ylabel("Best-balance label")
    ax.grid(False)
    ax.set_title("APCS confusion matrix (test)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, outdir, "fig5_confusion_matrix")


def fig_rq2(corr, outdir):
    c = corr[corr.target_rate == 0.5].pivot(index="feature", columns="method",
                                             values="rho")
    c = c.reindex(index=["token_count", "structural_complexity",
                         "fragmentation_ratio", "morphological_density"],
                  columns=["llmlingua2", "random_deletion", "llmlingua_qwen"])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", DIVERGING[::-1])
    im = ax.imshow(c.values, cmap=cmap, vmin=-0.8, vmax=0.8, aspect="auto")
    for (i, j), v in np.ndenumerate(c.values):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=INK,
                fontsize=9)
    short = {"llmlingua2": "LLMLingua-2", "random_deletion": "Random\ndeletion",
             "llmlingua_qwen": "LLMLingua\n(Qwen)"}
    ax.set_xticks(range(3), [short[m] for m in c.columns], fontsize=9)
    ax.set_yticks(range(4), ["token count", "structure", "fragmentation",
                             "morphology"])
    ax.grid(False)
    ax.set_title("Prompt features vs output fidelity\n"
                 "Spearman ρ, target rate 0.5, dev set", fontsize=10.5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, outdir, "fig6_rq2_features")


def fig_sweep(sw, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    shares = sorted(sw.share.unique())
    for ax, col, title in [(axes[0], "T1", "Calibrated 'do not compress below' threshold"),
                           (axes[1], "accuracy", "Rule accuracy on the mix")]:
        data = [sw[sw.share == s][col] for s in shares]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": INK})
        for b in bp["boxes"]:
            b.set(facecolor=BLUE_SEQ[1], edgecolor=BLUE_SEQ[4])
        ax.set_xticks(range(1, len(shares) + 1), [f"{s:.0%}" for s in shares])
        ax.set_xlabel("Synthetic share of the dev mix")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("tokens")
    fig.suptitle("Synthetic-share sweep (800-prompt mixes, 200 repeats each)",
                 fontweight="bold", color=INK, fontsize=11)
    save(fig, outdir, "fig7_synthetic_share_sweep")


def fig_09b(a9, outdir):
    rows = [(k.split("@")[0], v) for k, v in a9["per_category_rate"].items()
            if k.endswith("@0.5")]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    y = np.arange(len(rows))[::-1]
    for yy, (cat, v) in zip(y, rows):
        lo, hi = v["ci95_band_adjusted"]
        ax.plot([lo, hi], [yy, yy], color=SERIES["llmlingua2"], lw=2)
        ax.plot(v["diff_band_adjusted"], yy, "o", color=SERIES["llmlingua2"], ms=7)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Synthetic − sourced output F1 (length-band adjusted, 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Do synthetic prompts behave differently?\n"
                 "LLMLingua-2 at rate 0.5, dev set", fontsize=10.5)
    save(fig, outdir, "fig8_synthetic_vs_sourced")


def tables(an, ap, outdir):
    lines = ["# Exp 10 — thesis tables (auto-generated)\n"]
    if ap:
        p = ap["primary_arabert"]
        lines.append(f"## Table A — Policies on the held-out test set "
                     f"(n = {ap['n_test']}, τ = {p['tau']}, AraBERT)\n")
        lines.append("| Policy | Accuracy | Acc. (extended labels) | Mean TCR | "
                     "Mean out-F1 | Below τ | Cost saving vs none | Pareto hit |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for n, v in p["test_policies"].items():
            lines.append(f"| {n} | {v['accuracy']:.3f} | "
                         f"{v['accuracy_extended']:.3f} | {v['mean_tcr']:.3f} | "
                         f"{v['mean_out_f1']:.3f} | {v['violation_rate']:.1%} | "
                         f"{v['saving_vs_noop_pct']:.1f}% | {v['pareto_hit']:.3f} |")
        lines.append(f"\nAPCS accuracy 95% CI: {p['apcs_accuracy_ci95']}. "
                     f"Rule: {p['rule']}.\n")
        lines.append("## Table B — APCS vs each baseline (paired)\n")
        lines.append("| Baseline | Label set | Accuracy diff | 95% CI | McNemar p |")
        lines.append("|---|---|---|---|---|")
        for n, v in p["apcs_vs_baselines"].items():
            lines.append(f"| {n} | {v['label_set']} | {v['acc_diff']:+.3f} | "
                         f"{v['ci95']} | {v['mcnemar_p']:.2g} |")
        lines.append("\nAlways-LLMLingua baselines are compared on the extended "
                     "label set (which admits LLMLingua candidates); under the "
                     "primary labels they could never be correct.\n")
        s = ap["sensitivity_mbert"]
        lines.append(f"\nmBERT sensitivity: τ = {s['tau']}, rule {s['rule']}, "
                     f"test accuracy {s['test_policies']['APCS']['accuracy']:.3f}, "
                     f"beats all baselines: {s['apcs_beats_all_baselines']}.\n")
    if an:
        lines.append("## Table C — RQ1 method comparison (dev, output-level)\n")
        lines.append("| Comparison | n | A | B | Diff | 95% CI | Wilcoxon p |")
        lines.append("|---|---|---|---|---|---|---|")
        for sect in ["rq1_output_level", "rq1_prompt_level",
                     "llmlingua2_vs_llmlingua1_like_for_like"]:
            for k, v in an.get(sect, {}).items():
                if "diff" in v:
                    lines.append(f"| {sect}: {k} | {v['n']} | {v['a_mean']:.3f} | "
                                 f"{v['b_mean']:.3f} | {v['diff']:+.3f} | "
                                 f"{v['ci95']} | {v['wilcoxon_p']:.2g} |")
    (outdir / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("  wrote tables.md")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--indir", default=str(RESULTS))
    ap_.add_argument("--syndir", default=str(SYN))
    args = ap_.parse_args()
    indir, syndir = Path(args.indir), Path(args.syndir)
    outdir = indir / "figures"
    outdir.mkdir(exist_ok=True)

    def maybe(path, loader):
        return loader(path) if path.exists() else None

    res = maybe(indir / "final_results.csv", pd.read_csv)
    an = maybe(indir / "analysis_10.json", lambda p: json.loads(p.read_text()))
    ap = maybe(indir / "apcs_final.json", lambda p: json.loads(p.read_text()))
    ceil = maybe(indir / "ceiling.csv", pd.read_csv)
    corr = maybe(indir / "rq2_correlations.csv", pd.read_csv)
    sw = maybe(syndir / "sweep_09a_raw.csv", pd.read_csv)
    a9 = maybe(syndir / "analysis_09b.json", lambda p: json.loads(p.read_text()))
    tau = an["tau"]["arabert"]["adopted"] if an else 0.70

    if res is not None:
        fig_tradeoff(res, tau, outdir)
    if an:
        fig_flip(an, outdir)
    if ceil is not None:
        fig_ceiling(ceil, tau, outdir)
    if ap:
        fig_apcs(ap, outdir)
        fig_confusion(ap, outdir)
    if corr is not None:
        fig_rq2(corr, outdir)
    if sw is not None:
        fig_sweep(sw, outdir)
    if a9:
        fig_09b(a9, outdir)
    tables(an, ap, outdir)
    print(f"figures in {outdir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
