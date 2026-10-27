"""APCS demo: a Streamlit interface for walking through the artefact.

A presentation layer over the evaluated `apcs` package (not part of it):
recommendations come from the same frozen selectors, and the evidence tab
replays stored exam results, so no API key or API call is needed.

Run from the repository root:  streamlit run demo/app.py
"""

import html
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apcs"))
from apcs import APCSSelector, __version__  # noqa: E402
from apcs.features import LENGTH_MARKERS, arabic_letter_share  # noqa: E402
from apcs.selector import CATEGORIES  # noqa: E402

EXAM = ROOT / "experiments" / "11_fresh_exam" / "results"
PRICE_IN, PRICE_OUT = 0.20, 1.20          # US$ per million tokens (gpt-5.6-luna)
CL = tiktoken.get_encoding("cl100k_base")  # APCS measurement unit
O2 = tiktoken.get_encoding("o200k_base")   # tokeniser that billed the evaluated model
SELECTORS = {"v1": "APCS 1.0.0 (token-count rule)",
             "v2": "APCS-v2 (decision tree, best balance)",
             "cost": "APCS-cost (decision tree, cheapest faithful)"}
METHOD_LABEL = {"none": "No compression", "llmlingua2": "LLMLingua-2",
                "llmlingua2_protected": "LLMLingua-2, length instruction protected",
                "llmlingua": "LLMLingua (Qwen2.5-0.5B)", "noop": "No compression",
                "llmlingua_qwen": "LLMLingua (Qwen2.5-0.5B)"}
HIGHLIGHTS = ["exam-qa-043", "exam-qa-016", "exam-qa-030", "exam-qa-096", "exam-inst-045"]

st.set_page_config(page_title="APCS: Arabic prompt compression selector", page_icon="🗜️",
                   layout="wide")
st.markdown("""
<style>
textarea, .ar { unicode-bidi: plaintext; text-align: start; }
.ar { font-size: 1.08rem; line-height: 1.9; padding: .7rem .9rem; border-radius: .5rem;
      background: rgba(127,127,127,.09); white-space: pre-wrap; }
mark.len { background: #ffd666; color: #000; padding: 0 .15rem; border-radius: .2rem; }
.chips { line-height: 2.4; }
.chip { display: inline-block; padding: .05rem .45rem; margin: .12rem; border-radius: .3rem;
        font-size: 1.15rem; border: 1px solid rgba(127,127,127,.35); }
.chip.c0 { background: rgba(66,133,244,.18); } .chip.c1 { background: rgba(52,168,83,.18); }
.chip.c2 { background: rgba(251,188,5,.22); } .chip.c3 { background: rgba(234,67,53,.16); }
.badge { display: inline-block; padding: .1rem .5rem; border-radius: 1rem; font-size: .85rem; }
.badge.ok { background: rgba(52,168,83,.2); } .badge.bad { background: rgba(234,67,53,.2); }
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- helpers
@st.cache_resource
def selector(name):
    return APCSSelector(name)


@st.cache_data
def load_prompts():
    out = {}
    for f in ("AraPromptBench_v2.json", "AraPromptBench_exam.json"):
        d = json.loads((ROOT / f).read_text(encoding="utf-8"))
        for it in (d["prompts"] if isinstance(d, dict) else d):
            out[it["id"]] = it
    return out


@st.cache_data
def load_exam():
    er = pd.read_csv(EXAM / "exam_results.csv")
    er["cand"] = er.method + "@" + er.target_rate.astype(str)
    feats = pd.read_csv(EXAM / "exam_features.csv").set_index("prompt_id")
    return er, feats


def compression_available():
    try:
        import llmlingua  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_data(show_spinner=False)
def compress(prompt, method, rate):
    """Same code path as APCSSelector.compress, for any method and rate."""
    if method == "none":
        return prompt
    from apcs.protect import compress_protected
    from apcs.selector import _compressor
    if method == "llmlingua":
        return _compressor("Qwen/Qwen2.5-0.5B", False).compress_prompt(prompt, rate=rate)["compressed_prompt"]
    pc = _compressor("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", True)
    if method == "llmlingua2_protected":
        return compress_protected(pc, prompt, rate)
    return pc.compress_prompt(prompt, rate=rate)["compressed_prompt"]


def marked(text):
    """HTML with the length-instruction markers highlighted."""
    return LENGTH_MARKERS.sub(lambda m: f'<mark class="len">{m.group(0)}</mark>', html.escape(text))


def box(text, highlight=True):
    st.markdown(f'<div class="ar">{marked(text) if highlight else html.escape(text)}</div>',
                unsafe_allow_html=True)


def length_badge(original, compressed):
    if not LENGTH_MARKERS.search(original):
        return ""
    if LENGTH_MARKERS.search(compressed):
        return '<span class="badge ok">length instruction kept</span>'
    return '<span class="badge bad">length instruction deleted</span>'


def pieces(text, enc):
    out = []
    for t in enc.encode(text):
        b = enc.decode_single_token_bytes(t)
        try:
            out.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            out.append("⟨" + b.hex(" ") + "⟩")
    return out


def chips(text, enc):
    rtl = arabic_letter_share(text) > 0.5 if text.strip() else False
    spans = "".join(f'<span class="chip c{i % 4}">{html.escape(p) if p.strip() else "␣"}</span>'
                    for i, p in enumerate(pieces(text, enc)))
    st.markdown(f'<div class="chips" dir="{"rtl" if rtl else "ltr"}">{spans}</div>',
                unsafe_allow_html=True)


def rec_label(rec):
    return METHOD_LABEL[rec.method] + ("" if rec.method == "none" else f" @ rate {rec.rate:g}")


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🗜️ APCS")
    st.caption(f"Arabic-Aware Prompt Compression Selector · package v{__version__}")
    st.markdown(
        "Recommends, for each Arabic prompt, whether and how hard to compress it before "
        "sending it to an LLM.\n\n"
        "- Selectors calibrated on 800 AraPromptBench dev prompts, τ = 0.65\n"
        "- Evaluated once on a pre-registered 400-prompt exam\n"
        "- Evaluated model: gpt-5.6-luna, output priced 6× input")
    st.info("Deployment notes: route factual QA to no compression, and recalibrate when the "
            "model, the price ratio or τ changes.", icon="ℹ️")
    st.caption("Demo interface only: it calls the evaluated package and replays stored results; "
               "no API key is used.")

tab_tok, tab_rec, tab_ev, tab_batch = st.tabs(
    ["1 · Why Arabic costs more", "2 · Recommend & compress", "3 · Evidence from the exam", "4 · Batch"])

# --------------------------------------------------------------------------- 1 tokens
with tab_tok:
    st.subheader("The tokeniser tax")
    st.write("LLM APIs bill per token. Tokenisers trained mostly on English split Arabic words into "
             "more pieces, and Arabic attaches conjunctions, prepositions and pronouns to the word itself.")
    c1, c2 = st.columns(2)
    ar = c1.text_input("Arabic", "وبالكتاب")
    en = c2.text_input("English equivalent", "and with the book")
    for label, enc, note in (("cl100k_base", CL, "the APCS measurement unit"),
                             ("o200k_base", O2, "the tokeniser that billed the evaluated model")):
        st.markdown(f"**{label}** · {note}")
        a, b = st.columns(2)
        with a:
            chips(ar, enc)
            st.caption(f"{len(enc.encode(ar))} tokens for {len(ar.split())} word(s)")
        with b:
            chips(en, enc)
            st.caption(f"{len(enc.encode(en))} tokens for {len(en.split())} word(s)")
    st.caption("Across the 1,000 AraPromptBench prompts: 4.1 tokens per Arabic word under cl100k_base, "
               "1.9 under o200k_base. Newer tokenisers narrow the gap; compression attacks it from the "
               "other side, by sending fewer tokens.")

# --------------------------------------------------------------------------- 2 recommend
PROMPTS = load_prompts()
PRESETS = {
    "Short instruction (exam-inst-001)": (PROMPTS["exam-inst-001"]["prompt"], "instruction"),
    "Summary with a length instruction (v2-summ-051, Section 4.3)": (PROMPTS["v2-summ-051"]["prompt"],
                                                                     "summarisation"),
    "Long QA passage (exam-qa-030)": (PROMPTS["exam-qa-030"]["prompt"], "qa"),
    "Creative brief (exam-creat-001)": (PROMPTS["exam-creat-001"]["prompt"], "creative"),
    "English text (fail-safe)": ("Summarise the following article in three sentences: the city council "
                                 "approved the new transport plan after a long debate.", "summarisation"),
    "Encoding damage (fail-safe)": ("لخص النص التالي: ��� شهدت الأسواق ارتفاعا �� "
                                    "في أسعار الطاقة", "summarisation"),
}

with tab_rec:
    st.subheader("Recommend a compression strategy")
    pick = st.selectbox("Example prompt", ["(type your own)"] + list(PRESETS))
    if pick != "(type your own)" and st.session_state.get("_preset") != pick:
        st.session_state["prompt"], st.session_state["category"] = PRESETS[pick]
        st.session_state["_preset"] = pick
    prompt = st.text_area("Arabic prompt", key="prompt", height=170)
    c1, c2 = st.columns([2, 1])
    sel_name = c1.radio("Selector", list(SELECTORS), format_func=SELECTORS.get, horizontal=True, index=1)
    category = c2.selectbox("Task category", CATEGORIES, key="category",
                            disabled=sel_name == "v1", help="The tree selectors need the task category.")

    if prompt.strip():
        try:
            rec = selector(sel_name).recommend(prompt, None if sel_name == "v1" else category)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        f = rec.features
        st.markdown("### " + ("➡️ " if rec.method != "none" else "⏸️ ") + rec_label(rec))
        if rec.rule.startswith("fallback"):
            st.warning("Fail-safe: " + rec.rule.removeprefix("fallback: "), icon="🛡️")
        else:
            st.markdown("**Why:** the rule that fired")
            st.code(rec.rule, language=None)
        m = st.columns(6)
        m[0].metric("Tokens", f.token_count, help="cl100k_base, the APCS measurement unit")
        m[1].metric("Billed tokens", len(O2.encode(prompt)),
                    help="o200k_base, the tokeniser that billed the evaluated model (before message overhead)")
        m[2].metric("Words", f.word_count)
        m[3].metric("Tokens/word", f"{f.fragmentation_ratio:.2f}", help="Fragmentation ratio (cl100k_base)")
        m[4].metric("Structure", f.structural_complexity,
                    help="Line breaks, colons, quotes, brackets, bullets and enumerations")
        m[5].metric("Length instr.", "yes" if f.has_length_instruction else "no",
                    help="Contains an Arabic answer-length marker such as بإيجاز or فقرة واحدة")

        with st.expander("Compare all three selectors on this prompt"):
            rows = []
            for s in SELECTORS:
                r = selector(s).recommend(prompt, None if s == "v1" else category)
                rows.append({"Selector": SELECTORS[s], "Recommendation": rec_label(r), "Rule": r.rule})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("APCS-v2 maximises token reduction within the fidelity threshold; APCS-cost "
                       "minimises the measured bill, so it protects length instructions.")

        st.markdown("#### Apply the recommendation")
        if not compression_available():
            st.info("Install the compression extra to apply recommendations: "
                    "`pip install -e apcs[compression]`")
        elif st.button("Compress", type="primary", disabled=rec.method == "none"):
            st.session_state["compressed_for"] = (prompt, rec.method, rec.rate)
        if st.session_state.get("compressed_for") == (prompt, rec.method, rec.rate) and rec.method != "none":
            with st.spinner("Compressing (the model loads once, then stays in memory)…"):
                out = compress(prompt, rec.method, rec.rate)
            a, b = st.columns(2)
            with a:
                st.markdown(f"**Original** · {len(CL.encode(prompt))} tokens")
                box(prompt)
            with b:
                n = len(CL.encode(out))
                st.markdown(f"**{rec_label(rec)}** · {n} tokens "
                            f"({1 - n / len(CL.encode(prompt)):.0%} fewer) "
                            + length_badge(prompt, out), unsafe_allow_html=True)
                box(out)
            if f.has_length_instruction and rec.method in ("llmlingua2", "llmlingua2_protected"):
                other = "llmlingua2" if rec.method == "llmlingua2_protected" else "llmlingua2_protected"
                with st.expander(f"Same rate with {METHOD_LABEL[other]}"):
                    alt = compress(prompt, other, rec.rate)
                    st.markdown(f"{len(CL.encode(alt))} tokens " + length_badge(prompt, alt),
                                unsafe_allow_html=True)
                    box(alt)
            st.caption("Highlighted words are length-instruction markers. Deleting them lets answers "
                       "grow, which can raise the total bill even though the prompt got shorter (tab 3).")

# --------------------------------------------------------------------------- 3 evidence
with tab_ev:
    er, feats = load_exam()
    st.subheader("Fewer input tokens, bigger bill")
    st.write("Stored results from the fresh 400-prompt exam (real answers and billed tokens from "
             "gpt-5.6-luna). Nothing here calls the API.")

    li = feats.index[feats.has_length_instruction == 1]
    cost = er.pivot_table(index="prompt_id", columns="cand", values="cost_usd")
    def change(ids, cand):
        return 100 * (cost.loc[ids, cand].sum() / cost.loc[ids, "noop@1.0"].sum() - 1)

    agg = pd.DataFrame([{"Target rate": f"{r:g}",
                         "Plain LLMLingua-2": change(li, f"llmlingua2@{r}"),
                         "Length-protected": change(li, f"llmlingua2_protected@{r}"),
                         "Length-protected, all 400 prompts": change(cost.index, f"llmlingua2_protected@{r}")}
                        for r in (0.7, 0.5, 0.3)])
    st.markdown("**Total cost change against sending the uncompressed prompt, 180 prompts with a length "
                "instruction** (+ = costs more)")
    st.bar_chart(agg.set_index("Target rate")[["Plain LLMLingua-2", "Length-protected"]], stack=False,
                 y_label="% change in total bill", height=300)
    st.dataframe(agg.style.format({c: "{:+.1f}%" for c in agg.columns[1:]}), hide_index=True,
                 width="stretch")
    st.caption("Protection turns the penalty into a saving on prompts that carry a length instruction; "
               "across all prompts it still costs a little more than not compressing.")

    st.markdown("---")
    st.markdown("**Look at individual prompts**")
    c1, c2, c3 = st.columns([1.3, 2, 1])
    scope = c1.radio("Show", ["Highlighted examples", "Any prompt with a length instruction"])
    ids = HIGHLIGHTS if scope.startswith("High") else sorted(li)
    pid = c2.selectbox("Prompt", ids)
    rate = c3.selectbox("Target rate", [0.5, 0.3, 0.7])
    if scope.startswith("High"):
        st.caption("Highlighted examples were chosen because they show the mechanism clearly; the chart "
                   "above gives the average effect.")
    sub = er[er.prompt_id == pid].set_index("cand")
    original = sub.loc["noop@1.0", "compressed"]
    st.markdown(f"**Original prompt** · {pid} · {feats.loc[pid, 'task_category']}")
    box(original)
    cols = st.columns(3)
    variants = [("noop@1.0", "Uncompressed"), (f"llmlingua2@{rate}", f"LLMLingua-2 @ {rate:g}"),
                (f"llmlingua2_protected@{rate}", f"Protected @ {rate:g}")]
    base_cost = sub.loc["noop@1.0", "cost_usd"]
    for col, (cand, title) in zip(cols, variants):
        r = sub.loc[cand]
        with col:
            st.markdown(f"**{title}** " + (length_badge(original, r.compressed) if cand != "noop@1.0" else ""),
                        unsafe_allow_html=True)
            if cand != "noop@1.0":
                box(r.compressed)
            k = st.columns(2)
            k[0].metric("Input tokens", int(r.resp_prompt_tokens), help="Billed by the API")
            k[1].metric("Answer", int(r.resp_completion_tokens), help="Billed answer tokens")
            k = st.columns(2)
            k[0].metric("US$/1M calls", f"{r.cost_usd * 1e6:.1f}",
                        None if cand == "noop@1.0" else f"{100 * (r.cost_usd / base_cost - 1):+.0f}%",
                        delta_color="inverse")
            k[1].metric("Fidelity", f"{r.out_f1_arabert:.3f}", help="AraBERT BERTScore F1 against the uncompressed answer")
            with st.expander("Model's answer"):
                box(str(r.response), highlight=False)

# --------------------------------------------------------------------------- 4 batch
with tab_batch:
    st.subheader("Batch recommendations")
    st.write("Upload a CSV with a `prompt` column and, for the tree selectors, a `category` column "
             f"({', '.join(CATEGORIES)}).")
    sample = pd.DataFrame([{"prompt": p, "category": c} for p, c in PRESETS.values()])
    st.download_button("Download a sample CSV", sample.to_csv(index=False).encode("utf-8-sig"),
                       "apcs_sample_prompts.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    bsel = st.radio("Selector ", list(SELECTORS), format_func=SELECTORS.get, horizontal=True, index=2)
    if up is not None:
        df = pd.read_csv(up)
        if "prompt" not in df.columns:
            st.error("The file needs a `prompt` column.")
        else:
            out = []
            for _, row in df.iterrows():
                cat = row.get("category") if "category" in df.columns else None
                try:
                    r = selector(bsel).recommend(str(row["prompt"]), None if bsel == "v1" else cat)
                    out.append({"method": r.method, "rate": r.rate, "rule": r.rule,
                                "tokens_cl100k": r.features.token_count})
                except ValueError as e:
                    out.append({"method": "error", "rate": None, "rule": str(e), "tokens_cl100k": None})
            res = pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)
            st.dataframe(res, width="stretch", hide_index=True)
            counts = res.method.value_counts()
            st.caption(" · ".join(f"{METHOD_LABEL.get(k, k)}: {v}" for k, v in counts.items()))
            buf = io.BytesIO()
            res.to_csv(buf, index=False, encoding="utf-8-sig")
            st.download_button("Download recommendations", buf.getvalue(), "apcs_recommendations.csv",
                               "text/csv")
