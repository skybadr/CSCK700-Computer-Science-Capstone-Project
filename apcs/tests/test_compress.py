import sys
import types
import unicodedata

import pytest

from apcs import APCSSelector
from apcs import selector as sel
from apcs.protect import compress_protected

LONG = " ".join(["شهدت الأسواق ارتفاعا في أسعار الطاقة خلال الربع الأخير"] * 40)
BRIEF = "أجب بإيجاز: " + LONG + " السؤال: ماذا شهدت الأسواق؟"


class FakeCompressor:
    """Stands in for llmlingua.PromptCompressor: keeps the first share of words."""
    loads = 0

    def __init__(self, model_name, use_llmlingua2=False, device_map="cuda"):
        FakeCompressor.loads += 1
        FakeCompressor.device = device_map

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


def test_compressor_falls_back_to_cpu(fake_llmlingua, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)))
    sel._compressor("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", True)
    assert FakeCompressor.device == "cpu"


@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_protected_preserves_canonically_equivalent_instruction(form):
    """Feature detection and application agree without changing the text."""
    from apcs.features import extract_features

    instruction = unicodedata.normalize(form, "أجب بإيجاز:")
    text = instruction + " " + LONG
    assert extract_features(text).has_length_instruction

    class ReplacingCompressor:
        def compress_prompt(self, chunk, rate):
            return {"compressed_prompt": "[compressed context]"}

    out = compress_protected(ReplacingCompressor(), text, 0.3)
    assert out.startswith(instruction)
    assert "[compressed context]" in out
    assert len(out) < len(text)
