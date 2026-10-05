import json
from pathlib import Path

import pytest

from apcs import APCSSelector
from apcs.selector import walk_tree

SHORT = "اشرح مفهوم الذكاء الاصطناعي بأسلوب مبسط."
LONG_QA = ("اقرأ النص التالي ثم أجب عن السؤال: "
           + " ".join(["تأسست المدينة في القرن الثامن وكانت مركزا للتجارة"] * 30)
           + " السؤال: متى تأسست المدينة؟")
BRIEF = ("لخص النص التالي في ثلاث جمل: "
         + " ".join(["شهدت الأسواق ارتفاعا في أسعار الطاقة خلال الربع الأخير"] * 20))

EXAM = Path(__file__).resolve().parents[2] / "AraPromptBench_exam.json"
EVAL = (Path(__file__).resolve().parents[2] / "experiments" / "11_fresh_exam"
        / "results" / "exam_features.csv")


def test_v2_needs_category():
    with pytest.raises(ValueError):
        APCSSelector("v2").recommend(SHORT)


def test_v2_short_creative_not_compressed():
    rec = APCSSelector("v2").recommend(SHORT, category="creative")
    assert rec.method == "none" and "cat_creative > 0.5" in rec.rule


def test_v2_long_qa_compresses():
    rec = APCSSelector("v2").recommend(LONG_QA, category="qa")
    assert rec.method == "llmlingua2" and rec.rate in (0.5, 0.7)


def test_cost_protects_length_instruction():
    rec = APCSSelector("cost").recommend(BRIEF, category="summarisation")
    assert rec.features.has_length_instruction
    assert rec.method == "llmlingua2_protected" and rec.rate == 0.3


def test_walk_tree_missing_value_goes_left():
    tree = {"feature": "morphological_density", "threshold": 1.7,
            "le": {"leaf": "noop@1.0"}, "gt": {"leaf": "llmlingua2@0.5"}}
    assert walk_tree(tree, {"morphological_density": None})[0] == "noop@1.0"


@pytest.mark.skipif(not EXAM.exists() or not EVAL.exists(),
                    reason="needs the project's exam data")
@pytest.mark.parametrize("name", ["v2", "cost"])
def test_package_reproduces_frozen_exam_choices(name):
    """The shipped trees must choose exactly what the pre-registered exam
    evaluation chose, from the package's own feature extraction."""
    import pandas as pd
    feats = pd.read_csv(EVAL).set_index("prompt_id")
    prompts = json.loads(EXAM.read_text(encoding="utf-8"))["prompts"]
    frozen = json.loads((Path(__file__).resolve().parents[2] / "experiments"
                         / "11_fresh_exam" / "results" / "selectors"
                         / f"apcs_{name}.json").read_text(encoding="utf-8"))
    sel = APCSSelector(name)
    for p in prompts:
        row = feats.loc[p["id"]].to_dict()
        expected = walk_tree(frozen["tree"], row)[0]
        rec = sel.recommend(p["prompt"], category=p["category"])
        got = f"{'noop' if rec.method == 'none' else {'llmlingua': 'llmlingua_qwen'}.get(rec.method, rec.method)}@{rec.rate}"
        assert got == expected, p["id"]
