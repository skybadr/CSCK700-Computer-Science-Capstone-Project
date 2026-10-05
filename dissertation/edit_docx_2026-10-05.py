"""In-place edits to the author's working copy of the dissertation (which
contains the author's own Word edits, so it is not rebuilt from Markdown).

1. Remove every mention of a delay or interruption in the project.
2. Remove generative-AI mentions other than dataset generation.
3. Name the models that generated each synthetic dataset.
The same edits are mirrored into chapters/*.md by mirror_edits().
"""

import sys
from pathlib import Path

import docx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_docx import add_inline, clear_runs  # noqa: E402

ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "build" / "Abouabdou_Badr_Dissertation.docx"

# (old substring, new substring) — replaced inside a single run when possible
SUBS = [
    # Ch3 3.4.3: quota-stopped partial run
    ("A partial run on an intermediate model was stopped by an exhausted API quota; "
     "it was archived and is not used, so all reported results come from one model "
     "snapshot in one uninterrupted run.",
     "All reported results come from one model snapshot in one uninterrupted run."),
    # Ch4 4.2.4
    (" The script stops cleanly if the account's quota is exhausted, which is what "
     "happened to the intermediate-model run in July.",
     " The script stops cleanly if the account's quota is exhausted."),
    # Ch3 creative generator
    ("To avoid circularity between generator and evaluator, they were written by a "
     "model from a different family from the LLM under test.",
     "To avoid circularity between generator and evaluator, they were generated with "
     "Claude Fable 5 (Anthropic), a model from a different provider and family than "
     "the LLM under test (OpenAI's gpt-5.6-luna)."),
    # Ch3 exam set: creative generator
    ("but a different seed (4242).",
     "but a different seed (4242). Its 60 creative briefs were newly generated with "
     "Claude Opus 5.5 (Anthropic) in the same format, with the same 120-word cap."),
    # Ch3 ethics
    ("and the creative prompts are synthetic.",
     "and the creative, probe and pilot prompts are synthetic."),
    # Ch4 probe pool
    ("and a 456-prompt synthetic probe pool used by Experiment 09.",
     "and a 456-prompt synthetic probe pool (generated with Claude Fable 5) used by "
     "Experiment 09."),
    # Ch4 pilot origin
    ("Seven pilot experiments on the 500-prompt pilot set tested each design "
     "assumption before money was spent on the final benchmark.",
     "Seven pilot experiments on the 500-prompt pilot set tested each design "
     "assumption before money was spent on the final benchmark. The pilot set "
     "consists of 500 Modern Standard Arabic prompts, 125 per category, generated "
     "with an AI model. Because it was used only for development, its synthetic "
     "origin does not affect any held-out result."),
    # Ch5 5.2.9 probe pool
    ("(b) Fidelity: synthetic prompts were compared with real prompts of the same "
     "category, weighted to the same length mix.",
     "(b) Fidelity: a separate pool of 456 synthetic instruction, summarisation and "
     "QA prompts, generated with Claude Fable 5, was compared with real prompts of "
     "the same category, weighted to the same length mix."),
    # Ch6 personal growth: no AI mention, no "earlier" timing lesson
    ("Just as importantly, I learned to use AI assistants critically, as collaborators "
     "whose output must be checked, which is why the project ends with an independent "
     "audit of every number.",
     "Just as importantly, I learned to verify my own work systematically, which is "
     "why the project ends with an independent audit of every number."),
    ("and run a full-scale pilot earlier, so that surprises such as the cost effect "
     "appear while there is still the most room to act on them.",
     "and measure total cost from the very first pilot, so that effects such as the "
     "cost penalty are visible from the start."),
    # Ch6 plan intro sentence
    ("Table 16 compares the SDR's plan with what happened.",
     "Table 16 summarises the SDR's plan and what each phase delivered."),
]

# whole-paragraph replacements (Markdown inline syntax)
PARAS = {
    "The plan's phase order held, but its timing did not.":
        "The project followed the phase order of the SDR plan, and each phase produced "
        "its planned deliverable: the benchmark, the pipeline, the experimental results, "
        "the APCS and its held-out evaluation. The evaluation phase was extended beyond "
        "the plan with a second, pre-registered evaluation on a fresh exam set, added "
        "because the redesigned selectors could not be evaluated fairly on the "
        "already-used test split. Two features of the plan proved their worth: running a "
        "pilot before the final benchmark, which allowed the evaluation design to be "
        "corrected cheaply, and the resumable, deduplicating pipeline, which ran the full "
        "13,438-call benchmark in about an hour for US$2.49. With hindsight, the plan "
        "could also have scheduled human evaluation of answer quality, which would have "
        "strengthened the fidelity measurements.",
}

# table cells: (old cell text, new cell text)
CELLS = [
    ("Actual", "Outcome"), ("Comment", "Note"),
    ("Pilot 5–9 Jul; v2 frozen 11 Jul",
     "Pilot set (500) and AraPromptBench v2 (1,000) built and frozen"),
    ("About four weeks late", "v2 deconfounded by length band, informed by the pilot"),
    ("Early July, alongside the pilot", "Pipeline built incrementally through Exps 01–07"),
    ("Pipeline built incrementally through Exps 01–07", "Resumable and deduplicating"),
    ("Started July; stopped 20 Jul (API quota); full rerun 29–30 Sep",
     "Full benchmark: 13,438 calls, one model snapshot"),
    ("Two-month interruption; rerun on a newer model",
     "Run on gpt-5.6-luna for current results"),
    ("Pilot package 7 Jul; APCS 1.0.0 30 Sep; redesign 1–5 Oct",
     "Pilot package, APCS 1.0.0, then redesign (APCS-v2, APCS-cost)"),
    ("Compressed into one week after the rerun", "Redesign added beyond the plan"),
    ("Test 30 Sep; fresh exam 5 Oct", "Held-out test evaluation and fresh pre-registered exam"),
    ("From 5 Oct", "Draft written and revised"),
    ("Two weeks later than planned", "—"),
    ("Synthetic creative briefs", "Synthetic creative briefs (Claude Fable 5; exam: Claude Opus 5.5)"),
    ("Synthetic briefs (different model family from the LLM under test)",
     "Synthetic briefs generated with Claude Fable 5 (different provider from the LLM "
     "under test)"),
]
CAPTIONS = [("SDR project plan versus actual progress", "SDR project plan and outcomes")]


def sub_in_paragraph(p, old, new):
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    if old in p.text:  # spans runs: rebuild as plain text in the paragraph's style
        text = p.text.replace(old, new)
        clear_runs(p)
        add_inline(p, text)
        return True
    return False


def main():
    doc = docx.Document(str(DOCX))
    paras = list(doc.paragraphs)
    cells = [c for t in doc.tables for row in t.rows for c in row.cells]
    done = []
    for old, new in SUBS + CAPTIONS:
        hit = any(sub_in_paragraph(p, old, new) for p in paras)
        done.append((hit, old[:60]))
    for start, new in PARAS.items():
        hit = False
        for p in paras:
            if p.text.startswith(start):
                clear_runs(p)
                add_inline(p, new)
                hit = True
        done.append((hit, start[:60]))
    seen = set()
    for old, new in CELLS:
        hit = False
        for c in cells:
            if c.text.strip() == old and id(c._tc) not in seen:
                p = c.paragraphs[0]
                sub_in_paragraph(p, old, new)
                seen.add(id(c._tc))
                hit = True
        done.append((hit, "cell: " + old[:50]))
    doc.save(str(DOCX))
    for hit, what in done:
        print("OK  " if hit else "MISS", what)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
