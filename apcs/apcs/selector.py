"""APCS decision engine: ordered guarded rules over extracted features.

Rules are data (JSON), calibrated empirically on AraPromptBench
(Experiment 06). The shipped default was derived on the 500-prompt pilot
(dev-only) and will be re-calibrated on the final dataset's dev split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .features import PromptFeatures, extract_features


@dataclass(frozen=True)
class Recommendation:
    method: str          # "none" | "llmlingua2"
    rate: float          # target keep-rate (1.0 for "none")
    rule: str            # human-readable guard that fired
    features: PromptFeatures
    rules_version: str

    def as_dict(self) -> dict:
        d = {"method": self.method, "rate": self.rate, "rule": self.rule,
             "rules_version": self.rules_version}
        d["features"] = self.features.as_dict()
        return d


class APCSSelector:
    """Rule-based Arabic prompt-compression selector.

    >>> APCSSelector().recommend("...").method
    'llmlingua2'
    """

    def __init__(self, rules_path: str | Path | None = None):
        if rules_path is None:
            src = resources.files("apcs").joinpath("rules_default.json")
            cfg = json.loads(src.read_text(encoding="utf-8"))
        else:
            cfg = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        self.version = cfg["version"]
        self.tau = cfg["fidelity_threshold_tau"]
        t = cfg["thresholds"]
        self.t1, self.t2 = t["T1"], t["T2"]
        self.r_mid, self.r_long = t["r_mid"], t["r_long"]

    def recommend(self, prompt: str) -> Recommendation:
        f = extract_features(prompt)
        if f.token_count < self.t1:
            method, rate = "none", 1.0
            rule = f"token_count {f.token_count} < {self.t1}"
        elif f.token_count >= self.t2:
            method, rate = "llmlingua2", self.r_long
            rule = f"token_count {f.token_count} >= {self.t2}"
        else:
            method, rate = "llmlingua2", self.r_mid
            rule = (f"{self.t1} <= token_count {f.token_count} < {self.t2}")
        return Recommendation(method=method, rate=rate, rule=rule,
                              features=f, rules_version=self.version)

    def compress(self, prompt: str) -> str:
        """Convenience: recommend and apply (requires 'compression' extra)."""
        rec = self.recommend(prompt)
        if rec.method == "none":
            return prompt
        from llmlingua import PromptCompressor  # lazy heavy import
        pc = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True)
        return pc.compress_prompt(prompt, rate=rec.rate)["compressed_prompt"]
