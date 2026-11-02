"""Exp 10 — Feature extraction for all 1,456 final prompts (v2 + probe).

Same definitions as Exp 05 / the APCS package (apcs.features), so RQ2 on the
final data is directly comparable with the pilot. Morphological density via
Farasa in standalone mode (interactive mode corrupts Arabic on Windows —
Exp 05 finding). Features are prompt properties, not outcomes: computing them
for test prompts leaks nothing; RQ2 analysis itself uses dev rows only.

Requires Java on PATH. Output: results/features_final.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd

from apcs.features import density_from_segmented, extract_features

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
RESULTS = ROOT / "results"


def load_prompts():
    v2 = json.loads((PROJECT / "AraPromptBench_v2.json").read_text(encoding="utf-8"))
    probe = json.loads((ROOT.parent / "09_synthetic_sensitivity" / "probe_pool.json")
                       .read_text(encoding="utf-8"))
    rows = [(p["id"], p["prompt"], p["category"]) for p in v2["prompts"]]
    rows += [(p["id"], p["prompt"], p["category"]) for p in probe["prompts"]]
    return rows


def farasa_density(texts, chunk=100):
    from farasa.segmenter import FarasaSegmenter
    seg = FarasaSegmenter(interactive=False)
    flat = [" ".join(t.split()) for t in texts]
    out = []
    for i in range(0, len(flat), chunk):
        part = flat[i:i + chunk]
        lines = seg.segment("\n".join(part)).splitlines()
        if len(lines) != len(part):
            print(f"  chunk {i}: line mismatch, per-line fallback", flush=True)
            lines = [seg.segment(t) for t in part]
        out.extend(lines)
        print(f"  farasa {min(i + chunk, len(flat))}/{len(flat)}", flush=True)
    return [density_from_segmented(line) for line in out]


def main():
    prompts = load_prompts()
    rows = []
    for pid, text, cat in prompts:
        f = extract_features(text).as_dict()
        rows.append(dict(prompt_id=pid, task_category=cat, **f))
    df = pd.DataFrame(rows).drop(columns=["morphological_density"])
    df["morphological_density"] = farasa_density([t for _, t, _ in prompts])
    df.to_csv(RESULTS / "features_final.csv", index=False, encoding="utf-8-sig")
    print(df.groupby("task_category")[
        ["token_count", "fragmentation_ratio", "structural_complexity",
         "morphological_density"]].mean().round(2).to_string())
    print(f"nulls: {int(df.isna().sum().sum())} | saved features_final.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
