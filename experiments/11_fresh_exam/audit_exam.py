"""Independent overlap audit of the exam set (run after build_exam.py).

Checks exact text overlap with pilot / v2 / probe, duplicates within the
exam, record-id overlap with v2, and partial overlap: any 12-word stretch of
non-template text shared with v2. All counts must be zero.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
sys.path.insert(0, str(ROOT))
from build_exam import TEMPLATE_SH, norm, shingles  # noqa: E402


def main():
    load = lambda f: json.loads(f.read_text(encoding="utf-8"))["prompts"]
    ex = load(PROJECT / "AraPromptBench_exam.json")
    v2 = load(PROJECT / "AraPromptBench_v2.json")
    others = {"pilot": load(PROJECT / "AraPromptBench_dataset.json"), "v2": v2,
              "probe": load(ROOT.parent / "09_synthetic_sensitivity" / "probe_pool.json")}
    e = [norm(p["prompt"]) for p in ex]
    out = {f"exact_overlap_{k}": sum(x in {norm(p["prompt"]) for p in v}
                                     for x in e) for k, v in others.items()}
    out["duplicates_within_exam"] = len(e) - len(set(e))
    ids = {(p["source"], str(p["source_id"])) for p in v2}
    out["record_id_overlap_v2"] = sum((p["source"], p["source_id"]) in ids for p in ex)
    v2sh = set().union(*[shingles(p["prompt"]) for p in v2]) - TEMPLATE_SH
    out["partial_overlap_v2_12word"] = sum(
        bool((shingles(p["prompt"]) - TEMPLATE_SH) & v2sh) for p in ex)
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "overlap_audit.json").write_text(json.dumps(out, indent=2))
    sys.exit(0 if not any(out.values()) else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
