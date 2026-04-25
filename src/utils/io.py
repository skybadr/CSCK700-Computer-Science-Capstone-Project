"""I/O helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def load_prompts(path: str | Path) -> dict[str, Any]:
    """Load the AraPromptBench JSON dataset."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_results_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dict rows to a UTF-8 CSV.

    The first row's keys define the header. All rows must share the same keys.
    """
    rows_list = list(rows)
    if not rows_list:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows_list[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)
