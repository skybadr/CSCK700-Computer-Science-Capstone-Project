"""Final-revision edits, applied in place to the author's working .docx
(backup written first). Mirrors the same edits into chapters/*.md."""

import copy
import re
import shutil
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_docx import add_inline, clear_runs  # noqa: E402

ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "build" / "Abouabdou_Badr_Dissertation.docx"

SUBS = [
    # novelty hedges
    ("No Arabic prompt compression benchmark existed when this project began.",
     "To the author's knowledge, no Arabic prompt compression benchmark existed when this "
     "project began."),
    ("These findings have not been connected. Specifically:",
     "These findings have not been connected. Specifically, to the best of the author's "
     "knowledge, based on searches of the ACL Anthology and Google Scholar (October 2026):"),
    ("First, it provides the ", "First, it provides what is, to the author's knowledge, the "),
    (" for the first time,", ","),
    ("This is a new finding with practical weight.",
     "To the author's knowledge, this is a new finding, and it has practical weight."),
    # commit count / experiment count
    ("Git and a private GitHub repository recorded each milestone; with one commit per "
     "milestone (17 commits on the main branch at the time of writing).",
     "Git and a GitHub repository recorded each milestone, with one commit per milestone."),
    ("A chronological experiment log indexes all eleven experiments (Appendix E).",
     "A chronological experiment log indexes all eleven experiments and one supplementary "
     "measurement (Appendix E)."),
    # chance level
    ("which makes this a hard four-way choice in which the best fixed strategy can reach at "
     "most 29%.",
     "which makes this a hard four-way choice in which the best fixed strategy can reach at "
     "most 29% (chance level is 25%)."),
    (" APCS-v2 was the most accurate policy and significantly better than every alternative.",
     " APCS-v2 was the most accurate policy and significantly better than every alternative. "
     "For reference, chance level for this four-way choice is 25%, and random selection "
     "scored 24.5%."),
    # qualitative example
    ("so it \"saves\" money only by destroying fidelity.",
     "so it \"saves\" money only by destroying fidelity. Appendix D gives a worked example: "
     "on one exam QA prompt, plain compression deleted \"answer briefly\", the answer grew from "
     "23 to 85 tokens and the call cost nearly three times as much, while the protected "
     "variant kept the answer short."),
    # latency claim in business guidance
    ("for example to fit a context window or reduce latency.",
     "for example to fit a context window. Compression adds about 30 ms per prompt on a GPU "
     "(Section 4.3), so it is not a way to reduce latency for prompts of this length."),
    # code availability and reproduction
    ("private, and access can be granted to examiners on request). ",
     "access is provided to the examiners, as the module guidance requires). "),
    ("[Author: confirm how the code will be made available, as the module guidance requires.]",
     ""),
    ("Reproducing the final results: install the package and requirements in a Python 3.11 "
     "virtual environment,",
     "Reproducing the final results: create a Python 3.11 virtual environment and run "
     "pip install -r requirements.txt (the top-level README gives step-by-step instructions),"),
    ("  01_scorer_selection/ ... 11_fresh_exam/   one folder per experiment:",
     "  01_scorer_selection/ ... 12_compression_overhead/   one folder per experiment:"),
]

NEW_AFTER = [
    # (anchor paragraph start, new paragraph markdown, template = anchor)
    ("Testing. The package has 18 unit tests.",
     "**Documentation.** The package README serves as the user manual, covering installation, "
     "the command-line and library interfaces, the three selectors and recalibration. An "
     "example notebook (`apcs/examples/apcs_example.ipynb`) walks through single and batch "
     "recommendations, applying a recommendation and custom calibration. Together they deliver "
     "the manual and example notebook promised in the proposal."),
    ("**Documentation.**", None),  # placeholder to keep order (handled below)
    ("Possible training-data exposure.",
     "**Latency not measured end to end.** The compressor's own time was measured (Section "
     "4.3), but API response latency was not, so no claim is made that compression makes "
     "requests faster."),
    ("AraPromptBench_dataset.json     pilot set",
     "README.md, requirements.txt    start here: overview, quick start and pinned dependencies"),
]
OVERHEAD = (
    "**Overhead.** A supplementary measurement (Experiment 12; 100 dev prompts, rate 0.5) timed "
    "each step on the project workstation. The APCS recommendation itself takes about 0.05 ms per "
    "prompt, so the selector adds no meaningful latency. Applying a recommendation does take "
    "time: LLMLingua-2 needs about 30 ms per prompt on the consumer GPU and about 0.4 s on CPU, "
    "because it runs a 560-million-parameter encoder; the protected variant costs the same. For "
    "prompts of a few hundred tokens, this is of the same order as, or larger than, the "
    "prompt-processing time that compression removes. Compression should therefore be justified "
    "by cost or context-window limits rather than by speed. End-to-end API latency was not "
    "measured.")


def sub(paras, old, new):
    for p in paras:
        for r in p.runs:
            if old in r.text:
                r.text = r.text.replace(old, new)
                return True
    for p in paras:
        if old in p.text:
            text = p.text.replace(old, new)
            clear_runs(p)
            add_inline(p, text)
            return True
    return False


def insert_after(p, markdown):
    new = copy.deepcopy(p._p)
    p._p.addnext(new)
    np_ = docx.text.paragraph.Paragraph(new, p._parent)
    clear_runs(np_)
    if p.style.name == "Normal":            # code-block line: keep monospace
        r = np_.add_run(markdown)
        r.font.name = p.runs[0].font.name if p.runs else "Consolas"
        r.font.size = p.runs[0].font.size if p.runs else None
    else:
        add_inline(np_, markdown)
    return np_


def edit_explog(doc):
    for t in doc.tables:
        hdr = [c.text.strip() for c in t.rows[0].cells]
        if hdr[:2] == ["#", "Date"]:
            for row in t.rows:                  # drop the Date column
                row._tr.remove(row.cells[1]._tc)
            grid = t._tbl.find(qn("w:tblGrid"))
            cols = grid.findall(qn("w:gridCol"))
            cols[2].set(qn("w:w"), str(int(cols[1].get(qn("w:w"))) + int(cols[2].get(qn("w:w")))))
            grid.remove(cols[1])
            last = t.rows[-1]._tr
            new = copy.deepcopy(last)
            last.addnext(new)
            vals = ["12", "Compression overhead (supplementary)",
                    "Selector ≈0.05 ms; LLMLingua-2 ≈30 ms (GPU), ≈0.4 s (CPU)"]
            for tc, v in zip(new.findall(qn("w:tc")), vals):
                ts = list(tc.iter(qn("w:t")))
                ts[0].text = v
                for extra in ts[1:]:
                    extra.text = ""
            return True
    return False


def main():
    try:
        open(DOCX, "r+b").close()
    except PermissionError:
        sys.exit("LOCKED: close the document in Word first")
    shutil.copy(DOCX, DOCX.with_name("Abouabdou_Badr_Dissertation.backup-before-final-revision.docx"))
    doc = docx.Document(str(DOCX))
    paras = list(doc.paragraphs)
    report = []
    for old, new in SUBS:
        report.append((sub(paras, old, new), old[:60]))
    for anchor, md in NEW_AFTER:
        if md is None:
            continue
        hits = [p for p in doc.paragraphs if p.text.startswith(anchor)]
        if hits and not any(p.text.startswith(re.sub(r"\*\*|`", "", md)[:30]) for p in doc.paragraphs):
            np_ = insert_after(hits[0], md)
            if anchor.startswith("Testing."):
                insert_after(np_, OVERHEAD)
        report.append((bool(hits), "insert after: " + anchor[:40]))
    report.append((edit_explog(doc), "experiment log: Date column removed, Exp 12 row added"))
    doc.save(str(DOCX))
    for ok, what in report:
        print("OK  " if ok else "MISS", what)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
