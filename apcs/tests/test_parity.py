"""Optional end-to-end check: the package's compression step reproduces the
compressed prompts that were actually evaluated on the fresh exam.

Needs the 'compression' extra, the LLMLingua-2 model and ideally a GPU (about
90 s for 2,400 compressions). Run with:  APCS_PARITY=1 python -m pytest tests/test_parity.py
"""
import csv
import os
from pathlib import Path

import pytest

RESULTS = (Path(__file__).resolve().parents[2] / "experiments" / "11_fresh_exam"
           / "results" / "exam_results.csv")

pytestmark = pytest.mark.skipif(os.environ.get("APCS_PARITY") != "1",
                                reason="set APCS_PARITY=1 to run the slow parity check")


def test_package_reproduces_stored_exam_compressions():
    from apcs.protect import compress_protected
    from apcs.selector import _compressor
    with RESULTS.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    original = {r["prompt_id"]: r["compressed"] for r in rows if r["method"] == "noop"}
    pc = _compressor("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", True)
    checked = 0
    for r in rows:
        if r["method"] not in ("llmlingua2", "llmlingua2_protected"):
            continue
        text, rate = original[r["prompt_id"]], float(r["target_rate"])
        out = (compress_protected(pc, text, rate) if r["method"] == "llmlingua2_protected"
               else pc.compress_prompt(text, rate=rate)["compressed_prompt"])
        assert out == r["compressed"], (r["prompt_id"], r["method"], rate)
        checked += 1
    assert checked == 2400
