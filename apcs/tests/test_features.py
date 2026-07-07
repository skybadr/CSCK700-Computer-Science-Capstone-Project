import pytest

from apcs.features import extract_features, structural_complexity

ARABIC_SHORT = "اشرح مفهوم الذكاء الاصطناعي بأسلوب مبسط."
ARABIC_STRUCTURED = "أولاً: اكتب قائمة.\n1. البند الأول\n2. البند الثاني\n«اقتباس»"


def test_basic_features():
    f = extract_features(ARABIC_SHORT)
    assert f.word_count == 6
    assert f.token_count > f.word_count  # Arabic fragments into subwords
    assert f.fragmentation_ratio == pytest.approx(
        f.token_count / f.word_count, abs=1e-3)
    assert f.character_length == len(ARABIC_SHORT)
    assert f.morphological_density is None  # off by default


def test_structural_complexity_counts_markers():
    assert structural_complexity(ARABIC_SHORT) == 0
    assert structural_complexity(ARABIC_STRUCTURED) >= 5  # \n x2, :, enum x2, «»


def test_empty_prompt_rejected():
    for bad in ["", "   ", None]:
        with pytest.raises((ValueError, TypeError)):
            extract_features(bad)


def test_deterministic():
    assert extract_features(ARABIC_SHORT) == extract_features(ARABIC_SHORT)
