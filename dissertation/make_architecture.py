"""Figure 1: APCS architecture (as built) and the calibration pipeline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED = "#1f2933", "#52606d"
BLUE, BLUE_L = "#2b6cb0", "#dbe8f6"
GREEN, GREEN_L, GREY_L = "#2f855a", "#e3f1e6", "#eef0f2"

fig, ax = plt.subplots(figsize=(11.5, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")


def box(x, y, w, h, title, body, fc=BLUE_L, ec=BLUE, fs=9.5, bs=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, lw=1.3,
                                boxstyle="round,pad=0.4,rounding_size=1.2"))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=fs,
            color=INK, weight="bold")
    ax.text(x + w / 2, y + h - 5.6, "\n".join(body), ha="center", va="top",
            fontsize=bs, color=MUTED, linespacing=1.35)


def arrow(x1, y1, x2, y2, label="", dashed=False, lx=0.0, ly=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color=INK, lw=1.1, linestyle="--" if dashed else "-"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7.8,
                color=MUTED, ha="center", va="center", style="italic")


ax.add_patch(FancyBboxPatch((1, 33), 98, 28, fc="white", ec=BLUE, lw=1.0,
                            boxstyle="round,pad=0.3,rounding_size=1.5"))
ax.text(3, 59.2, "APCS artefact (installable Python package, apcs 1.1.0)",
        fontsize=10, weight="bold", color=BLUE, va="top")
ax.add_patch(FancyBboxPatch((1, 1), 98, 28, fc="white", ec=MUTED, lw=1.0, ls="--",
                            boxstyle="round,pad=0.3,rounding_size=1.5"))
ax.text(3, 27.2, "Evaluation and calibration pipeline (experiments, not shipped)",
        fontsize=10, weight="bold", color=MUTED, va="top")

box(4, 37, 15, 17, "Input",
    ["Arabic prompt", "+ task category", "(v2 / cost only)"], fc=GREY_L, ec=MUTED)
box(24, 37, 20, 17, "Feature extractor",
    ["token count (cl100k)", "fragmentation ratio", "structural complexity",
     "length-instruction flag", "[morph. density: optional]"])
box(49, 37, 21, 17, "Decision engine",
    ["v1: token-count rules", "v2: decision tree", "cost: decision tree",
     "rules read from JSON"])
box(75, 37, 21, 17, "Recommendation",
    ["method + keep rate", "+ rule that fired", "(explainable)",
     "-> optional compress()"], fc=GREEN_L, ec=GREEN)
arrow(19.3, 45.5, 23.6, 45.5)
arrow(44.3, 45.5, 48.6, 45.5)
arrow(70.3, 45.5, 74.6, 45.5)

kw = dict(fc=GREY_L, ec=MUTED, fs=8.6, bs=7.2)
box(3, 5, 15, 17, "AraPromptBench",
    ["v2: 1,000 prompts", "800 dev / 200 test", "exam: 400 prompts"], **kw)
box(21, 5, 17, 17, "Compressors",
    ["LLMLingua-2", "LLMLingua-2-protected", "LLMLingua (Qwen)", "random deletion",
     "@ 0.7 / 0.5 / 0.3"], **kw)
box(41, 5, 14, 17, "LLM under test",
    ["gpt-5.6-luna", "temperature 0", "billed tokens", "per call"], **kw)
box(58, 5, 18, 17, "Evaluation",
    ["output BERTScore", "(AraBERT; mBERT check)", "noise-ceiling τ = 0.65",
     "measured cost", "QA correctness"], **kw)
box(79, 5, 18, 17, "Calibration",
    ["labels: best balance /", "cheapest faithful", "grid search;",
     "CART + 10-fold CV", "(dev only)"], **kw)
arrow(18.4, 13.5, 20.6, 13.5)
arrow(38.4, 13.5, 40.6, 13.5)
arrow(55.4, 13.5, 57.6, 13.5)
arrow(76.4, 13.5, 78.6, 13.5)
arrow(88, 22.4, 62, 36.6, "frozen rules / trees (JSON)", dashed=True, lx=8, ly=1.8)

fig.savefig("figures/fig_architecture.png", dpi=220, bbox_inches="tight")
