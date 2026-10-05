"""APCS decision engine.

Three selectors, all calibrated on the AraPromptBench v2 dev split (800
prompts) and frozen as data files:

- "v1"   (APCS 1.0.0, rules_default.json): ordered token-count rules
         (Experiment 10). Needs only the prompt.
- "v2"   (APCS-v2, selectors/apcs_v2.json): depth-4 decision tree over the
         feature vector and the task category, maximising best-balance
         accuracy (Experiment 11, H1/H2 supported on the fresh exam).
- "cost" (APCS-cost, selectors/apcs_cost.json): depth-3 tree choosing the
         cheapest strategy that keeps output fidelity above tau, including
         the length-protected LLMLingua-2 variant (Experiment 11, H3).

The tree selectors need the prompt's task category, one of
instruction / summarisation / qa / creative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .features import PromptFeatures, extract_features

CATEGORIES = ("instruction", "summarisation", "qa", "creative")
_TREE_FILES = {"v2": "apcs_v2.json", "cost": "apcs_cost.json"}
_METHOD_NAMES = {"noop": "none", "llmlingua2": "llmlingua2",
                 "llmlingua2_protected": "llmlingua2_protected",
                 "llmlingua_qwen": "llmlingua"}


@dataclass(frozen=True)
class Recommendation:
    method: str          # "none" | "llmlingua2" | "llmlingua2_protected" | "llmlingua"
    rate: float          # target keep-rate (1.0 for "none")
    rule: str            # human-readable guard(s) that fired
    features: PromptFeatures
    rules_version: str

    def as_dict(self) -> dict:
        d = {"method": self.method, "rate": self.rate, "rule": self.rule,
             "rules_version": self.rules_version}
        d["features"] = self.features.as_dict()
        return d


def _load(rules_path, package_file):
    if rules_path is not None:
        return json.loads(Path(rules_path).read_text(encoding="utf-8"))
    src = resources.files("apcs")
    for part in package_file.split("/"):
        src = src.joinpath(part)
    return json.loads(src.read_text(encoding="utf-8"))


def tree_inputs(f: PromptFeatures, category: str) -> dict:
    """Feature dictionary in the form the frozen trees were trained on."""
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    x = {"token_count": f.token_count,
         "structural_complexity": f.structural_complexity,
         "fragmentation_ratio": f.fragmentation_ratio,
         "morphological_density": f.morphological_density,
         "has_length_instruction": int(f.has_length_instruction)}
    x.update({f"cat_{c}": int(c == category) for c in CATEGORIES})
    return x


def walk_tree(node: dict, x: dict) -> tuple[str, list[str]]:
    """Return (leaf label, path of guards). A missing feature value (only
    morphological density, which is optional) follows the <= branch; in the
    shipped trees both of its branches lead to the same strategy."""
    path = []
    while "leaf" not in node:
        name, thr = node["feature"], node["threshold"]
        v = x[name]
        if v is None or v <= thr:
            path.append(f"{name} <= {thr:g}")
            node = node["le"]
        else:
            path.append(f"{name} > {thr:g}")
            node = node["gt"]
    return node["leaf"], path


class APCSSelector:
    """Arabic prompt-compression selector.

    >>> APCSSelector().recommend("...").method                    # APCS 1.0.0
    >>> APCSSelector("v2").recommend("...", category="qa").method  # APCS-v2
    """

    def __init__(self, selector: str = "v1", rules_path: str | Path | None = None):
        if selector not in ("v1", "v2", "cost"):
            raise ValueError("selector must be 'v1', 'v2' or 'cost'")
        self.selector = selector
        if selector == "v1":
            cfg = _load(rules_path, "rules_default.json")
            self.version = cfg["version"]
            self.tau = cfg["fidelity_threshold_tau"]
            t = cfg["thresholds"]
            self.t1, self.t2 = t["T1"], t["T2"]
            self.r_mid, self.r_long = t["r_mid"], t["r_long"]
        else:
            cfg = _load(rules_path, "selectors/" + _TREE_FILES[selector])
            self.version = cfg["name"]
            self.tau = cfg["tau"]
            self.tree = cfg["tree"]

    def recommend(self, prompt: str, category: str | None = None) -> Recommendation:
        f = extract_features(prompt)
        if self.selector == "v1":
            method, rate, rule = self._rules_v1(f)
        else:
            if category is None:
                raise ValueError(f"selector '{self.selector}' needs the task "
                                 f"category, one of {CATEGORIES}")
            leaf, path = walk_tree(self.tree, tree_inputs(f, category))
            name, r = leaf.split("@")
            method, rate, rule = _METHOD_NAMES[name], float(r), " and ".join(path)
        return Recommendation(method=method, rate=rate, rule=rule,
                              features=f, rules_version=self.version)

    def _rules_v1(self, f):
        if f.token_count < self.t1:
            return "none", 1.0, f"token_count {f.token_count} < {self.t1}"
        if f.token_count >= self.t2:
            return ("llmlingua2", self.r_long,
                    f"token_count {f.token_count} >= {self.t2}")
        return ("llmlingua2", self.r_mid,
                f"{self.t1} <= token_count {f.token_count} < {self.t2}")

    def compress(self, prompt: str, category: str | None = None) -> str:
        """Recommend and apply (requires the 'compression' extra)."""
        rec = self.recommend(prompt, category)
        if rec.method == "none":
            return prompt
        from llmlingua import PromptCompressor  # lazy heavy import
        if rec.method == "llmlingua":
            pc = PromptCompressor(model_name="Qwen/Qwen2.5-0.5B")
            return pc.compress_prompt(prompt, rate=rec.rate)["compressed_prompt"]
        pc = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True)
        if rec.method == "llmlingua2_protected":
            from .protect import compress_protected
            return compress_protected(pc, prompt, rec.rate)
        return pc.compress_prompt(prompt, rate=rec.rate)["compressed_prompt"]
