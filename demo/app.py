"""APCS demo: one Arabic prompt's journey through the artefact.

A presentation layer over the evaluated `apcs` package (not part of it).
Recommendations and compression come from the package itself; the outcome
step replays stored exam results, so no API key or API call is needed.

Run from the repository root:  streamlit run demo/app.py
"""

import html
import importlib.util
import io
import json
import re
import sys
import threading
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import streamlit as st
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apcs"))
from apcs import APCSSelector, __version__  # noqa: E402
from apcs.features import LENGTH_MARKERS, arabic_letter_share, extract_features  # noqa: E402
from apcs.selector import CATEGORIES, tree_inputs, walk_tree  # noqa: E402

EXAM = ROOT / "experiments" / "11_fresh_exam" / "results"
LL2 = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
CL = tiktoken.get_encoding("cl100k_base")   # APCS measurement unit
O2 = tiktoken.get_encoding("o200k_base")    # tokeniser that billed the evaluated model
TAU = 0.65                                  # fidelity threshold (calibrated on repeat calls)

STEPS = ["Prompt", "Read", "Decide", "Compress", "Outcome"]
SELECTORS = {
    "v1": ("APCS 1.0.0", "token rule",
           "One rule on prompt length: under 80 tokens the prompt is left alone, 80–249 tokens "
           "keep about half, 250 or more keep about 30%."),
    "v2": ("APCS-v2", "best balance",
           "A small decision tree over length, layout, fragmentation and task category. It picks the "
           "strongest compression whose answers stayed faithful during calibration."),
    "cost": ("APCS-cost", "lowest bill",
             "A small decision tree trained on the measured bill, prompt and answer together. It picks "
             "the cheapest faithful strategy and can protect answer-length instructions."),
}
CATEGORY = {"instruction": "Instruction", "summarisation": "Summarisation",
            "qa": "Question answering", "creative": "Creative writing"}
SHORT_CATEGORY = {"instruction": "Instruction", "summarisation": "Summary", "qa": "QA",
                  "creative": "Creative"}
GLOSS = {"بإيجاز": "briefly", "موجز": "brief", "جملتين": "two sentences", "ثلاث جمل": "three sentences",
         "جملة واحدة": "one sentence", "فقرة واحدة": "one paragraph", "لا يتجاوز": "not exceeding",
         "كلمة": "words"}
STORED_METHOD = {"none": "noop", "llmlingua2": "llmlingua2",
                 "llmlingua2_protected": "llmlingua2_protected", "llmlingua": "llmlingua_qwen"}
EXAMPLES = {
    "“Answer briefly”": {
        "pid": "exam-qa-016",
        "gist": "Based on the following paragraph, answer briefly: the observable universe holds an "
                "estimated 100–200 billion galaxies … Question: how many galaxies are there?"},
    "Passage question": {
        "pid": "exam-qa-058",
        "gist": "Text: China, officially the People's Republic of China, is the world's most populous "
                "country … Based on the text above, answer: what is the People's Republic of China "
                "called in Chinese?"},
    "Three-sentence summary": {
        "pid": "exam-summ-051",
        "gist": "Summarise the following text in three sentences at most: Dubai: Microsoft Gulf "
                "announced that DoConnect won its “Best Innovation” award at the 2003 retail "
                "developer awards in Paris."},
    "Short request": {
        "pid": "exam-inst-001",
        "gist": "Write a speech about how technology changes our daily lives."},
    "English text": {
        "text": "Summarise the following article in three sentences: the city council approved the new "
                "transport plan after a long debate.",
        "category": "summarisation",
        "note": "Not Arabic, so outside the prompts APCS was calibrated on."},
    "Damaged text": {
        "text": "لخص النص التالي: ��� شهدت الأسواق ارتفاعا �� في أسعار الطاقة",
        "category": "summarisation",
        "gist": "Summarise the following text: [damaged characters] markets saw energy prices rise."},
    "Your own": {},
}

st.set_page_config(page_title="APCS · Arabic prompt compression", page_icon=":material/compress:",
                   layout="wide")
st.markdown("""
<style>
.block-container { max-width: 1120px; padding-top: 2.2rem; }
.overline { font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: #0F766E;
            font-weight: 600; }
.lead { color: #4B5563; font-size: 1.05rem; margin-top: -.6rem; }
textarea, .ar { unicode-bidi: plaintext; text-align: start; }
.card { border: 1px solid #E5E7EB; border-radius: .75rem; padding: .9rem 1.2rem; background: #FAFBFC;
        margin-bottom: .8rem; }
.ar { font-family: "Segoe UI", "Noto Naskh Arabic", "Noto Sans Arabic", Tahoma, sans-serif;
      font-size: 1.16rem; line-height: 2.05; }
.gist { color: #6B7280; font-size: .9rem; margin-top: .5rem; }
.meta { color: #9CA3AF; font-size: .8rem; margin-top: .25rem; }
.arw { font-family: "Segoe UI", "Noto Naskh Arabic", Tahoma, sans-serif; font-size: 1.15em; }
.cut { color: #A1A8B3; text-decoration: line-through; text-decoration-color: #B4BBC5; }
.punct { color: #A1A8B3; }
.mark { background: #FEF3C7; color: #92400E; border-radius: .25rem; padding: 0 .2rem; }
.mark.cut { background: #FEE2E2; color: #B91C1C; text-decoration-color: #B91C1C; }
.legend { font-size: .82rem; color: #6B7280; margin: -.3rem 0 .6rem; }
.legend > span { margin-right: 1.4rem; }
.rec { border-radius: .9rem; padding: 1.1rem 1.3rem; background: #F0FDFA; border: 1px solid #99F6E4; }
.rec .k { font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; color: #0F766E;
          font-weight: 600; }
.rec .v { font-size: 1.55rem; font-weight: 700; color: #134E4A; margin: .25rem 0 .1rem; }
.rec .r { font-size: 1.02rem; color: #115E59; }
.rec .n { color: #4B5563; margin-top: .6rem; font-size: .92rem; }
.rec.none { background: #F8FAFC; border-color: #E2E8F0; }
.rec.none .v { color: #334155; } .rec.none .r, .rec.none .k { color: #475569; }
.rec.safe { background: #FFFBEB; border-color: #FDE68A; }
.rec.safe .v { color: #92400E; } .rec.safe .r, .rec.safe .k { color: #A16207; }
.path { list-style: none; padding: 0; margin: 0; counter-reset: q; }
.path li { display: flex; justify-content: space-between; gap: 1rem; padding: .5rem 0;
           border-bottom: 1px dashed #E5E7EB; }
.path .q::before { counter-increment: q; content: counter(q); display: inline-block; width: 1.4rem;
                   height: 1.4rem; line-height: 1.4rem; margin-right: .6rem; border-radius: 50%;
                   background: #E6F4F1; color: #0F766E; font-size: .75rem; text-align: center;
                   font-weight: 600; }
.path .a { font-weight: 600; color: #0F766E; white-space: nowrap; }
.path li.leaf { border-bottom: none; font-weight: 600; color: #134E4A; }
.path li.leaf .q::before { content: "→"; background: none; font-size: 1rem; }
.chips { line-height: 2.4; }
.chip { display: inline-block; padding: .05rem .45rem; margin: .12rem; border-radius: .3rem;
        font-size: 1.12rem; border: 1px solid #D9DEE4; }
.chip.c0 { background: #E0F2FE; } .chip.c1 { background: #DCFCE7; }
.chip.c2 { background: #FEF9C3; } .chip.c3 { background: #FCE7F3; }
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data and models
@st.cache_resource
def selector(name):
    return APCSSelector(name)


@st.cache_resource   # read-only lookups: shared, not copied on every rerun
def exam_prompts():
    d = json.loads((ROOT / "AraPromptBench_exam.json").read_text(encoding="utf-8"))
    return {p["id"]: p for p in d["prompts"]}


@st.cache_resource
def exam_runs():
    """Stored exam calls, keyed by (prompt id, method, target rate)."""
    cols = ["prompt_id", "method", "target_rate", "compressed", "response", "resp_prompt_tokens",
            "resp_completion_tokens", "cost_usd", "out_f1_arabert"]
    er = pd.read_csv(EXAM / "exam_results.csv", usecols=cols)
    return {(r.prompt_id, r.method, float(r.target_rate)): r._asdict()
            for r in er.itertuples(index=False)}


def stored(pid, method, rate):
    return exam_runs().get((pid, STORED_METHOD[method], float(rate))) if pid else None


@st.cache_data
def exam_summary():
    ev = json.loads((EXAM / "exam_evaluation.json").read_text(encoding="utf-8"))["policies"]
    return {s: ev[name] for s, name in (("v1", "APCS-1.0.0"), ("v2", "APCS-v2"), ("cost", "APCS-cost"))}


@st.cache_data
def length_instruction_effect(rate=0.5):
    """Total-bill change on the exam prompts that carry a length instruction."""
    feats = pd.read_csv(EXAM / "exam_features.csv")
    ids = feats.prompt_id[feats.has_length_instruction == 1]
    runs = exam_runs()
    base = sum(runs[(i, "noop", 1.0)]["cost_usd"] for i in ids)
    return len(ids), *(100 * (sum(runs[(i, m, rate)]["cost_usd"] for i in ids) / base - 1)
                       for m in ("llmlingua2", "llmlingua2_protected"))


def compression_available():
    return importlib.util.find_spec("llmlingua") is not None   # without the slow import


@st.cache_resource(show_spinner=False)
def model_loader():
    """Start loading LLMLingua-2 in the background while the first steps are explained."""
    from apcs.selector import _compressor
    t = threading.Thread(target=_compressor, args=(LL2, True), daemon=True)
    t.start()
    return t


@st.cache_data(show_spinner=False, max_entries=64)
def run_compression(sel_name, prompt, category):
    model_loader().join()
    return selector(sel_name).compress(prompt, category)


# --------------------------------------------------------------------------- presentation helpers
_AR_MARKS = "ؐ-ًؚ-ٰٟۖ-ۭ"
_WORD = re.compile(rf"[\w{_AR_MARKS}]+")
_PIECE = re.compile(rf"[\w{_AR_MARKS}]+|\s+|[^\w\s{_AR_MARKS}]+")


def esc(text):
    return html.escape(text).replace("\n", "<br>")


def answer_html(text):
    """A model answer as HTML, with its Markdown bold, headings and bullets made readable."""
    lines = []
    for line in html.escape(text).split("\n"):
        line = re.sub(r"^\s*#{1,6}\s*(.*)$", r"<b>\1</b>", line)
        line = re.sub(r"^\s*[-*]\s+", "• ", line)
        lines.append(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line))
    return "<br>".join(lines)


def annotate(original, compressed=None):
    """The prompt as HTML, with answer-length markers highlighted and, given the
    compressed text, deleted words struck through. LLMLingua-2 keeps surviving
    words in their original order, so aligning the two word sequences recovers
    the deletions (punctuation is shown faded: the compressor drops most of it)."""
    pieces = [(m.start(), m.group()) for m in _PIECE.finditer(original)]
    words = [i for i, (_, p) in enumerate(pieces) if _WORD.fullmatch(p)]
    kept = set(words)
    if compressed is not None:
        kept = set()
        sm = SequenceMatcher(None, [pieces[i][1] for i in words], _WORD.findall(compressed),
                             autojunk=False)
        for a, _, n in sm.get_matching_blocks():
            kept.update(words[a:a + n])
    marks = [m.span() for m in LENGTH_MARKERS.finditer(original)]
    out = []
    for i, (start, p) in enumerate(pieces):
        if p.isspace():
            out.append("<br>" * min(p.count("\n"), 2) or " ")
            continue
        if _WORD.fullmatch(p):
            cls = [] if i in kept else ["cut"]
        else:
            cls = [] if compressed is None else ["punct"]
        if any(s < start + len(p) and start < e for s, e in marks):
            cls.append("mark")
        out.append(f'<span class="{" ".join(cls)}">{html.escape(p)}</span>' if cls else html.escape(p))
    return "".join(out)


def card(body_html, gist=None, meta=None, note=None):
    parts = f'<div class="ar">{body_html}</div>'
    if gist:
        parts += f'<div class="gist">English gist: {html.escape(gist)}</div>'
    if note:
        parts += f'<div class="gist">{html.escape(note)}</div>'
    if meta:
        parts += f'<div class="meta">{html.escape(meta)}</div>'
    st.markdown(f'<div class="card">{parts}</div>', unsafe_allow_html=True)


def chips(text, enc):
    spans = []
    for i, t in enumerate(enc.encode(text)):
        b = enc.decode_single_token_bytes(t)
        try:
            p = b.decode("utf-8")
        except UnicodeDecodeError:
            p = "◌"   # the token holds only part of a letter's bytes
        spans.append(f'<span class="chip c{i % 4}">{html.escape(p) if p.strip() else "␣"}</span>')
    rtl = arabic_letter_share(text) > 0.5
    st.markdown(f'<div class="chips" dir="{"rtl" if rtl else "ltr"}">{"".join(spans)}</div>',
                unsafe_allow_html=True)


def arw(word):
    """Arabic inside an English sentence, in the Arabic font."""
    return f'<span class="arw">{html.escape(word)}</span>'


def marker(text):
    """The first answer-length marker in the text (HTML), with an English gloss."""
    m = LENGTH_MARKERS.search(text)
    return f"«{arw(m.group())}» ({GLOSS.get(m.group(), 'length')})" if m else None


def strategy(method, rate):
    """Display name and keep share of a strategy."""
    if method == "none":
        return "No compression", "the whole prompt is sent"
    name = {"llmlingua2": "LLMLingua-2", "llmlingua2_protected": "LLMLingua-2, protected",
            "llmlingua": "LLMLingua"}[method]
    return name, f"keep about {rate:.0%} of the prompt"


def short(rec):
    name, _ = strategy(rec.method, rec.rate)
    return name if rec.method == "none" else f"{name}, keep {rec.rate:.0%}"


def phrase(guard, f):
    """One decision-tree guard as a yes/no question about this prompt."""
    name, op, thr = guard.split()
    yes = "Yes" if op == ">" else "No"
    t = float(thr)
    if name.startswith("cat_"):
        return f"{CATEGORY[name[4:]]}?", yes
    if name == "token_count":
        return f"Longer than {int(t)} tokens?", f"{yes} ({f.token_count})"
    if name == "fragmentation_ratio":
        return f"More than {t:.2f} tokens per word?", f"{yes} ({f.fragmentation_ratio:.2f})"
    if name == "structural_complexity":
        return f"More than {int(t)} layout marks?", f"{yes} ({f.structural_complexity})"
    if name == "has_length_instruction":
        return "Answer-length instruction?", yes
    return "Morphological density high?", "not measured (optional)"


def decision_path(sel_name, rec, prompt, category):
    if rec.rule.startswith("fallback"):
        if "U+FFFD" in rec.rule:
            return [("Text intact, without damaged characters?", "No")]
        return [("Text intact, without damaged characters?", "Yes"),
                ("At least half of the letters Arabic?", f"No ({arabic_letter_share(prompt):.0%})")]
    f = rec.features
    if sel_name == "v1":
        s, tc = selector("v1"), f.token_count
        steps = [(f"Shorter than {s.t1} tokens?", f"{'Yes' if tc < s.t1 else 'No'} ({tc})")]
        if tc >= s.t1:
            steps.append((f"{s.t2} tokens or more?", f"{'Yes' if tc >= s.t2 else 'No'} ({tc})"))
        return steps
    _, path = walk_tree(selector(sel_name).tree, tree_inputs(f, category))
    return [phrase(g, f) for g in path]


def delta(new, old):
    return None if new == old else f"{100 * (new / old - 1):+.0f}%"


def go(step):
    st.session_state.step = step


def load_example():
    ex = EXAMPLES[st.session_state.example]
    if "pid" in ex:
        st.session_state.category = exam_prompts()[ex["pid"]]["category"]
    elif "category" in ex:
        st.session_state.category = ex["category"]


def nav():
    i = STEPS.index(st.session_state.step)
    st.write("")
    left, _, right = st.columns([1, 3, 1.3])
    if i > 0:
        left.button("Back", icon=":material/arrow_back:", on_click=go, args=(STEPS[i - 1],),
                    width="stretch")
    if i < len(STEPS) - 1:
        right.button(f"Next: {STEPS[i + 1]}", icon=":material/arrow_forward:", icon_position="right",
                     type="primary", on_click=go, args=(STEPS[i + 1],), width="stretch")
    else:
        right.button("Start again", icon=":material/restart_alt:", on_click=go, args=(STEPS[0],),
                     width="stretch")


# --------------------------------------------------------------------------- state
for key, value in {"step": STEPS[0], "example": next(iter(EXAMPLES)), "own_prompt": "",
                   "category": "qa", "selector": "v2", "compressed": set()}.items():
    st.session_state.setdefault(key, value)
if compression_available():
    model_loader()

ex = EXAMPLES[st.session_state.example]
pid = ex.get("pid")
prompt = exam_prompts()[pid]["prompt"] if pid else ex.get("text", st.session_state.own_prompt)
category, sel = st.session_state.category, st.session_state.selector
try:
    feats = extract_features(prompt)
    recs = {s: selector(s).recommend(prompt, None if s == "v1" else category) for s in SELECTORS}
    rec = recs[sel]
except ValueError:
    feats = rec = None

# --------------------------------------------------------------------------- header
st.markdown(f'<div class="overline">IT artefact · apcs {__version__}</div>', unsafe_allow_html=True)
st.title("Arabic-Aware Prompt Compression Selector")
st.markdown('<p class="lead">Follow one Arabic prompt through APCS: read it, decide on a strategy, '
            'compress it, and see what happened to the answer and the bill.</p>', unsafe_allow_html=True)
st.segmented_control("Journey", STEPS, key="step", required=True, label_visibility="collapsed",
                     format_func=lambda s: f"{STEPS.index(s) + 1}  {s}", width="stretch")
step = st.session_state.step
st.write("")

# --------------------------------------------------------------------------- 1 prompt
if step == "Prompt":
    st.subheader("Choose a prompt")
    st.caption("APCS works on one prompt at a time. Pick an example from the held-out exam set, "
               "or type your own.")
    st.pills("Example", list(EXAMPLES), key="example", required=True, on_change=load_example,
             persist_state="page", label_visibility="collapsed")
    if not ex:
        st.text_area("Your Arabic prompt", key="own_prompt", height=180, persist_state="page",
                     placeholder="اكتب طلبك هنا…")
    else:
        meta = None
        if pid:
            meta = f"{pid} · source: {exam_prompts()[pid]['source']} · held-out exam set, never used to design APCS"
        card(esc(prompt), ex.get("gist"), meta, ex.get("note"))
    st.segmented_control("Task category", list(CATEGORY), key="category", format_func=CATEGORY.get,
                         required=True, persist_state="page")
    st.caption("The two decision-tree selectors use the task category, which the caller supplies.")
    nav()
    st.stop()

if rec is None:
    st.info("Enter a prompt in step 1 first.", icon=":material/edit_note:")
    nav()
    st.stop()

# --------------------------------------------------------------------------- 2 read
if step == "Read":
    st.subheader("Read the prompt")
    st.caption("APCS measures a few simple features. This takes milliseconds and needs no GPU.")
    card(annotate(prompt))
    f = feats
    m = st.columns(5)
    m[0].metric("Length", f"{f.token_count} tokens", border=True,
                help="Counted with cl100k_base, the fixed unit APCS uses for every prompt.")
    m[1].metric("Tokens per word", f"{f.fragmentation_ratio:.2f}", border=True,
                help="How finely the tokeniser splits each word (fragmentation).")
    m[2].metric("Layout marks", f.structural_complexity, border=True,
                help="Line breaks, colons, quotes, brackets, bullets and numbering.")
    m[3].metric("Length instruction", "Yes" if f.has_length_instruction else "No", border=True,
                help="An Arabic instruction that limits the answer, such as بإيجاز (briefly).")
    m[4].metric("Task", SHORT_CATEGORY[category], border=True, help="Supplied by the caller in step 1.")
    if f.has_length_instruction:
        st.markdown(f"**Length instruction found:** {marker(prompt)}, highlighted above. "
                    "If a compressor deletes it, the answer tends to get longer, and answer tokens "
                    "cost six times as much as prompt tokens.", unsafe_allow_html=True)
    st.caption(f"Script check: {arabic_letter_share(prompt):.0%} of the letters are Arabic. Below 50%, "
               "or with damaged characters, APCS declines to compress.")
    with st.expander("Why Arabic needs more tokens"):
        sample = " ".join(prompt.split()[:6])
        chips(sample, CL)
        st.caption(f"These {len(sample.split())} words are {len(CL.encode(sample))} tokens in cl100k_base "
                   f"and {len(O2.encode(sample))} in o200k_base, the tokeniser that billed the evaluated "
                   "model (◌ marks a token holding only part of a letter). Arabic also attaches "
                   "conjunctions, prepositions and pronouns to the word itself:")
        a, b = st.columns(2)
        with a:
            chips("وبالكتاب", CL)
            st.caption(f"{arw('وبالكتاب')}: one word, 6 tokens", unsafe_allow_html=True)
        with b:
            chips("and with the book", CL)
            st.caption("and with the book: four words, 4 tokens")
    nav()

# --------------------------------------------------------------------------- 3 decide
elif step == "Decide":
    st.subheader("Decide on a strategy")
    st.caption("A selector turns the features into a recommendation: whether to compress, how, and how hard.")
    st.segmented_control("Selector", list(SELECTORS), key="selector", required=True, persist_state="page",
                         format_func=lambda s: f"{SELECTORS[s][0]} · {SELECTORS[s][1]}")
    st.caption(SELECTORS[sel][2])
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        name, keep = strategy(rec.method, rec.rate)
        if rec.rule.startswith("fallback"):
            kind, note = "safe", "Outside the prompts APCS was calibrated on, so it declines to compress (fail-safe)."
        elif rec.method == "none":
            kind, note = "none", {
                "v1": "Under 80 tokens: too short for compression to be worth the risk.",
                "v2": "Too short, or already split into many tokens per word, for compression to pay off.",
                "cost": "On the measured bill, compressing prompts like this did not pay off during calibration.",
            }[sel]
        elif rec.method == "llmlingua2_protected":
            kind, note = "go", (f"The sentence carrying {marker(prompt)} is kept word for word; "
                                "the rest of the prompt is compressed.")
        else:
            kind, note = "go", "LLMLingua-2 scores every word and deletes the least informative ones."
        st.markdown(f'<div class="rec {kind}"><div class="k">{SELECTORS[sel][0]} recommends</div>'
                    f'<div class="v">{name}</div><div class="r">{keep}</div>'
                    f'<div class="n">{note}</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown("**How it decided**")
        items = "".join(f'<li><span class="q">{html.escape(q)}</span><span class="a">{html.escape(a)}</span></li>'
                        for q, a in decision_path(sel, rec, prompt, category))
        st.markdown(f'<ol class="path">{items}<li class="leaf"><span class="q">{html.escape(short(rec))}'
                    '</span></li></ol>', unsafe_allow_html=True)
    st.write("")
    st.caption("Other selectors on this prompt: " + " · ".join(
        f"{SELECTORS[s][0]} → {short(recs[s])}" for s in SELECTORS if s != sel))
    nav()

# --------------------------------------------------------------------------- 4 compress
elif step == "Compress":
    st.subheader("Compress")
    name, _ = strategy(rec.method, rec.rate)
    if rec.method == "none":
        st.info("Nothing to compress: the prompt goes to the model unchanged.", icon=":material/check_circle:")
        card(esc(prompt))
    elif not compression_available():
        st.info("Install the compression extra to run this step: `pip install -e apcs[compression]`")
    else:
        st.caption(f"{name} keeps about {rec.rate:.0%} of the prompt. Struck-through words are deleted, "
                   "so the model never sees them.")
        key = (sel, prompt, category)
        if key not in st.session_state.compressed:
            card(annotate(prompt))
            st.button(f"Compress with {name}", type="primary", icon=":material/compress:",
                      on_click=st.session_state.compressed.add, args=(key,))
        else:
            with st.spinner("Compressing (the model loads once, then stays in memory)…"):
                out = run_compression(sel, prompt, None if sel == "v1" else category)
            t0, t1 = len(CL.encode(prompt)), len(CL.encode(out))
            b0, b1 = len(O2.encode(prompt)) + 6, len(O2.encode(out)) + 6
            m = st.columns(3)
            m[0].metric("Tokens (cl100k_base)", f"{t0} → {t1}", delta(t1, t0), delta_color="inverse", border=True)
            m[1].metric("Billed prompt tokens", f"{b0} → {b1}", delta(b1, b0), delta_color="inverse",
                        border=True, help="o200k_base plus the 6-token message overhead billed by the evaluated model.")
            kept = "none in the prompt" if not LENGTH_MARKERS.search(prompt) else (
                "kept" if LENGTH_MARKERS.search(out) else "deleted")
            m[2].metric("Length instruction", kept, border=True)
            card(annotate(prompt, out))
            st.markdown('<div class="legend"><span><span class="cut">struck through</span>: deleted</span>'
                        '<span><span class="mark">highlighted</span>: answer-length instruction</span></div>',
                        unsafe_allow_html=True)
            with st.expander("The compressed prompt, as sent to the model"):
                card(esc(out))
            run = stored(pid, rec.method, rec.rate)
            if run is not None and run["compressed"] == out:
                st.caption(":material/verified: Identical to the compression evaluated in the exam: "
                           "same package code, same model.")
    nav()

# --------------------------------------------------------------------------- 5 outcome
elif step == "Outcome":
    st.subheader("Check the outcome")
    st.caption("A shorter prompt is only half the story: the answer, and its length, decide the rest of the bill.")
    base, run = stored(pid, "none", 1.0), stored(pid, rec.method, rec.rate)
    if run is not None:
        st.markdown(f"**Uncompressed prompt → {short(rec)}** ({SELECTORS[sel][0]})")
        in0, in1 = int(base["resp_prompt_tokens"]), int(run["resp_prompt_tokens"])
        a0, a1 = int(base["resp_completion_tokens"]), int(run["resp_completion_tokens"])
        k0, k1 = base["cost_usd"] * 1e6, run["cost_usd"] * 1e6
        fid = run["out_f1_arabert"]
        m = st.columns(4)
        m[0].metric("Prompt tokens", f"{in0} → {in1}", delta(in1, in0), delta_color="inverse", border=True)
        m[1].metric("Answer tokens", f"{a0} → {a1}", delta(a1, a0), delta_color="inverse", border=True)
        m[2].metric("Cost per million calls", f"\\${k0:.1f} → \\${k1:.1f}", delta(k1, k0), delta_color="inverse",
                    border=True, help=r"US\$0.20 per million prompt tokens and US\$1.20 per million answer tokens.")
        m[3].metric("Answer fidelity", f"{fid:.2f}", None if rec.method == "none" else f"{fid - TAU:+.2f} vs τ",
                    border=True, help=f"AraBERT BERTScore F1 against the answer to the uncompressed prompt; "
                                      f"τ = {TAU} was calibrated on repeat calls.")
        change = 100 * (k1 / k0 - 1)
        found = LENGTH_MARKERS.search(prompt)
        lost = GLOSS.get(found.group()) if found and not LENGTH_MARKERS.search(run["compressed"]) else None
        if rec.method == "none":
            st.info("The prompt was sent unchanged: same answer, same bill.", icon=":material/check_circle:")
        elif change < 0 and fid >= TAU:
            st.success(f"Cheaper by {-change:.0f}%, and the answer stayed faithful (fidelity {fid:.2f} ≥ τ {TAU}).",
                       icon=":material/check_circle:")
        elif change < 0:
            st.warning(f"Cheaper by {-change:.0f}%, but the answer drifted (fidelity {fid:.2f} < τ {TAU}).",
                       icon=":material/warning:")
        else:
            why = f"Compression deleted the length instruction (“{lost}”), and the" if lost else "The"
            st.error(f"Shorter prompt, bigger bill: {change:+.0f}%. {why} answer grew from {a0} to {a1} "
                     "tokens, which cost six times as much as prompt tokens.", icon=":material/trending_up:")
        with st.expander("Read both answers"):
            a, b = st.columns(2)
            for col, title, r in ((a, "Without APCS", base), (b, f"With {SELECTORS[sel][0]}", run)):
                with col:
                    st.markdown(f"**{title}** · {int(r['resp_completion_tokens'])} tokens")
                    with st.container(height=260, border=False):
                        card(answer_html(str(r["response"])))
        st.caption("Stored result from the pre-specified exam run (gpt-5.6-luna, temperature 0). "
                   "This demo replays it and makes no API calls.")
    elif rec.method == "none":
        st.info("The prompt is sent unchanged: same answer, same bill.", icon=":material/check_circle:")
    else:
        st.info("This prompt is not from the exam set, so there is no stored answer to show, and this demo "
                "does not call the model. The prompt gets shorter; the answer decides the rest of the bill.",
                icon=":material/info:")
        if feats.has_length_instruction and rec.method == "llmlingua2":
            n, plain, protected = length_instruction_effect(rec.rate)
            st.warning(f"This prompt has an answer-length instruction that plain compression may delete. On the "
                       f"{n} exam prompts with one, plain LLMLingua-2 keeping {rec.rate:.0%} changed the total "
                       f"bill by {plain:+.1f}%; the protected variant by {protected:+.1f}%.",
                       icon=":material/warning:")

    st.markdown("##### Across all 400 exam prompts")
    summary = exam_summary()
    rows = ["| Selector | Total bill vs no compression | Answers below τ |", "|---|---|---|"]
    for s, (label, *_rest) in SELECTORS.items():
        below = f"{100 * summary[s]['violation_rate']:.2f}".rstrip("0").rstrip(".")
        cells = [label, f"{summary[s]['cost_change_pct']:+.1f}%".replace("-", "−"), f"{below}%"]
        rows.append("| " + " | ".join(f"**{c}**" if s == sel else c for c in cells) + " |")
    st.markdown("\n".join(rows))
    st.caption("APCS-v2 also picked the best-balance strategy more often than any fixed strategy "
               "(38.5% vs 32.8%, Holm p = 0.028). Examples on this page were chosen to show the mechanism; "
               "the table gives the averages.")

    st.markdown("##### Use APCS in your own code")
    cat_arg = "" if sel == "v1" else f', category="{category}"'
    st.code(f'from apcs import APCSSelector\n\nselector = APCSSelector("{sel}")\n'
            f"rec = selector.recommend(prompt{cat_arg})\n"
            f"print(rec.method, rec.rate)   # {rec.method} {rec.rate:g}\n"
            f"compressed = selector.compress(prompt{cat_arg})   # needs apcs[compression]", language="python")
    st.code(f"apcs --selector {sel}" + ("" if sel == "v1" else f" --category {category}") + " --file prompt.txt",
            language="bash")
    with st.expander("Run APCS on a CSV of prompts"):
        st.caption("A `prompt` column and, for the tree selectors, a `category` column "
                   f"({', '.join(CATEGORIES)}).")
        up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
        if up is not None:
            df = pd.read_csv(up)
            if "prompt" not in df.columns:
                st.error("The file needs a `prompt` column.")
            else:
                out = []
                for _, row in df.iterrows():
                    try:
                        r = selector(sel).recommend(str(row["prompt"]),
                                                    None if sel == "v1" else row.get("category"))
                        out.append({"method": r.method, "rate": r.rate, "rule": r.rule})
                    except ValueError as e:
                        out.append({"method": "error", "rate": None, "rule": str(e)})
                res = pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)
                st.dataframe(res, hide_index=True, width="stretch")
                buf = io.BytesIO()
                res.to_csv(buf, index=False, encoding="utf-8-sig")
                st.download_button("Download recommendations", buf.getvalue(), "apcs_recommendations.csv",
                                   "text/csv")
    st.caption("Calibrated for gpt-5.6-luna, a 6:1 answer-to-prompt price ratio and τ = 0.65: recalibrate "
               "when any of these change. Where factual answers matter most, route QA prompts to no compression.")
    nav()
