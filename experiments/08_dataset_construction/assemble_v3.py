"""Assemble the combined AraPromptBench v3 release from frozen component sets.

Run from the repository root:
    python experiments/08_dataset_construction/assemble_v3.py
    python experiments/08_dataset_construction/assemble_v3.py --check

This release combines existing records. It does not rerun an experiment or
change any prompt, ID, source, licence, answer or evaluation partition.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "AraPromptBench_v3.json"
RELEASE_DATE = "2026-11-02"
COMPONENTS = (
    ("AraPromptBench_v2.json", 1000, {"dev": 800, "test": 200}),
    ("AraPromptBench_exam.json", 400, {"exam": 400}),
)


def normalise(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ً-ْـ]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(rows, field):
    return dict(sorted(Counter(row[field] for row in rows).items()))


def assemble():
    components = []
    prompts = []
    for filename, expected_size, expected_splits in COMPONENTS:
        path = PROJECT / filename
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
        rows = data["prompts"]
        require(len(rows) == expected_size, f"Unexpected size: {filename}")
        require(counts(rows, "split") == expected_splits,
                f"Unexpected original partitions: {filename}")
        for row in rows:
            require(all(row.get(key) for key in ("id", "prompt", "source", "license")),
                    f"Missing identity or source terms: {filename}")
        components.append({
            "file": filename,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "total_prompts": len(rows),
            "split_counts": counts(rows, "split"),
            "metadata": data["metadata"],
        })
        prompts.extend(rows)

    require(len(prompts) == 1400, "The combined release must contain 1,400 rows.")
    require(len({row["id"] for row in prompts}) == 1400, "Duplicate prompt IDs.")
    require(len({row["prompt"] for row in prompts}) == 1400, "Duplicate prompt text.")
    require(len({normalise(row["prompt"]) for row in prompts}) == 1400,
            "Duplicate prompts after the original builder's normalisation.")

    metadata = {
        "name": "AraPromptBench",
        "version": "3.0.0",
        "created": RELEASE_DATE,
        "release_kind": "Combined distribution of the original benchmark and fresh exam",
        "description": (
            "A 1,400-prompt release assembled after evaluation from AraPromptBench "
            "v2 (1,000 prompts) and AraPromptBench-exam (400 prompts). Original "
            "records and evaluation partitions are preserved. This is a combined "
            "distribution, not a new experimental run or a new random split."
        ),
        "language": "ar",
        "total_prompts": len(prompts),
        "category_counts": counts(prompts, "category"),
        "source_counts": counts(prompts, "source"),
        "split_counts": counts(prompts, "split"),
        "licence_counts": counts(prompts, "license"),
        "partitions": {
            "dev": {
                "count": 800,
                "component": "AraPromptBench_v2.json",
                "role": "Original development and selector-design partition",
            },
            "test": {
                "count": 200,
                "component": "AraPromptBench_v2.json",
                "role": "Original held-out test used in Experiment 10",
            },
            "exam": {
                "count": 400,
                "component": "AraPromptBench_exam.json",
                "role": "Fresh exam used in Experiment 11 after designs were fixed",
            },
        },
        "evaluation_use": (
            "Published results refer to the original component partitions. "
            "Keep exam records separate from development when reproducing those "
            "results; v3 as a whole is not an unseen evaluation set."
        ),
        "licensing": {
            "scope": "Per-record source terms; no blanket dataset licence",
            "provenance": "DATA_PROVENANCE.md",
            "reviewed_source_labels": "2026-11-01",
            "note": (
                "Every source and license field is inherited unchanged. Source "
                "terms, including unresolved EASC and ARCD terms and XL-Sum's "
                "CC BY-NC-SA 4.0 label, continue to apply. The APCS package's "
                "MIT licence does not license third-party dataset content."
            ),
        },
        "synthetic_prompt_provenance": [
            {
                "component": "AraPromptBench_v2.json",
                "count": 150,
                "generator": "Claude Fable 5 (Anthropic)",
                "construction_records": [
                    f"experiments/08_dataset_construction/creative_prompts_part{i}.json"
                    for i in (1, 2, 3)
                ],
            },
            {
                "component": "AraPromptBench_exam.json",
                "count": 60,
                "generator": "Claude Opus 5.5 (Anthropic)",
                "generated": "2026-10-04",
                "construction_records": [
                    "experiments/11_fresh_exam/creative_exam.json"
                ],
            },
        ],
        "excluded_sets": [
            {
                "file": "AraPromptBench_dataset.json",
                "count": 500,
                "role": "Earlier pilot, distributed separately",
            },
            {
                "file": "experiments/09_synthetic_sensitivity/probe_pool.json",
                "count": 456,
                "role": "Synthetic sensitivity pool, distributed separately",
            },
        ],
        "components": components,
    }
    return {"metadata": metadata, "prompts": prompts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Check the saved v3 file against its original components.")
    args = parser.parse_args()
    expected = assemble()
    if args.check:
        actual = json.loads(OUTPUT.read_text(encoding="utf-8"))
        require(actual == expected,
                "Saved v3 does not match the original component records and metadata.")
        action = "Verified"
    else:
        OUTPUT.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        action = "Wrote"
    print(f"{action} {OUTPUT.name}: 1,400 unique prompts "
          "(800 dev / 200 test / 400 exam); source terms preserved.")


if __name__ == "__main__":
    main()
