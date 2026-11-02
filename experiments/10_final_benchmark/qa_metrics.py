"""Correctness metrics for extractive-QA prompts (TyDi-QA / ARCD gold answers).

Model answers are free-form sentences, so exact match is too strict. Two
lenient, standard measures, both after Arabic normalisation (diacritics and
tatweel removed, alef/ya/ta-marbuta unified, punctuation stripped):
  contains  1 if any gold answer appears verbatim inside the answer
  recall    share of the best gold answer's words present in the answer
"""

import re
import unicodedata

_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_PUNCT = re.compile(r"[^\w\s]")


def normalise(text):
    t = unicodedata.normalize("NFC", text or "")
    t = _DIACRITICS.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = _PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def contains(answer, golds):
    a = f" {normalise(answer)} "
    return int(any(f" {normalise(g)} " in a for g in golds if normalise(g)))


def recall(answer, golds):
    words = set(normalise(answer).split())
    best = 0.0
    for g in golds:
        gw = normalise(g).split()
        if gw:
            best = max(best, sum(w in words for w in gw) / len(gw))
    return best
