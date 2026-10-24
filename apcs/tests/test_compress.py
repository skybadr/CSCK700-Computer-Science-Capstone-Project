import sys
import types

import pytest

from apcs import APCSSelector
from apcs import selector as sel
from apcs.protect import compress_protected

LONG = " ".join(["شهدت الأسواق ارتفاعا في أسعار الطاقة خلال الربع الأخير"] * 40)
BRIEF = "أجب بإيجاز: " + LONG + " السؤال: ماذا شهدت الأسواق؟"


class FakeCompressor:
    """Stands in for llmlingua.PromptCompressor: keeps the first share of words."""
    loads = 0

    def __init__(self, model_name, use_llmlingua2=False):
        FakeCompressor.loads += 1

    def compress_prompt(self, text, rate):
        words = text.split()
        return {"compressed_prompt": " ".join(words[:max(1, int(len(words) * rate))])}


@pytest.fixture
def fake_llmlingua(monkeypatch):
    monkeypatch.setitem(sys.modules, "llmlingua",
                        types.SimpleNamespace(PromptCompressor=FakeCompressor))
    FakeCompressor.loads = 0
    sel._compressor.cache_clear()
    yield
    sel._compressor.cache_clear()


def test_compress_loads_model_once(fake_llmlingua):
    s = APCSSelector()
    first, second = s.compress(LONG), s.compress(LONG)
    assert first == second and len(first) < len(LONG)
    assert FakeCompressor.loads == 1


def test_no_compression_needs_no_model(fake_llmlingua):
    short = "اشرح مفهوم الذكاء الاصطناعي بأسلوب مبسط."
    assert APCSSelector().compress(short) == short
    assert FakeCompressor.loads == 0


def test_protected_keeps_length_instruction():
    out = compress_protected(FakeCompressor("x", True), BRIEF, 0.3)
    assert out.startswith("أجب بإيجاز:") and len(out) < len(BRIEF)
    # without a marker the text is compressed exactly as plain LLMLingua-2 would
    pc = FakeCompressor("x", True)
    assert compress_protected(pc, LONG, 0.3) == pc.compress_prompt(LONG, 0.3)["compressed_prompt"]
