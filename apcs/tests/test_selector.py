import json
from pathlib import Path

import pytest

from apcs import APCSSelector

SHORT = "اشرح مفهوم الذكاء الاصطناعي بأسلوب مبسط."  # far below T1 tokens
# ~repeat to exceed thresholds
MID = " ".join(["يوضح الجدول متوسط الطلبات الشهرية لكل فرع من الفروع"] * 8)
LONG = " ".join(["يتناول هذا المقال تطور الاقتصاد الرقمي في المنطقة العربية"] * 40)


@pytest.fixture(scope="module")
def sel():
    return APCSSelector()


def test_rules_loaded_from_package(sel):
    assert sel.version
    assert sel.t1 < sel.t2


def test_short_prompt_not_compressed(sel):
    rec = sel.recommend(SHORT)
    assert rec.method == "none" and rec.rate == 1.0
    assert str(rec.features.token_count) in rec.rule


def test_mid_prompt_gets_llmlingua2(sel):
    rec = sel.recommend(MID)
    assert sel.t1 <= rec.features.token_count
    assert rec.method == "llmlingua2"
    assert rec.rate in (0.3, 0.5, 0.7)


def test_long_prompt_gets_llmlingua2(sel):
    rec = sel.recommend(LONG)
    assert rec.features.token_count >= sel.t2
    assert rec.method == "llmlingua2"


def test_custom_rules_file(tmp_path: Path, sel):
    custom = {"version": "test", "fidelity_threshold_tau": 0.7,
              "thresholds": {"T1": 5, "T2": 10, "r_mid": 0.7, "r_long": 0.3}}
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(custom), encoding="utf-8")
    s = APCSSelector(rules_path=p)
    assert s.recommend(SHORT).method == "llmlingua2"  # T1=5 -> compresses


def test_recommendation_serialisable(sel):
    d = sel.recommend(MID).as_dict()
    assert json.dumps(d, ensure_ascii=False)
    assert d["features"]["token_count"] > 0
