"""Command-line interface: `apcs "<arabic prompt>"` or `apcs --file prompt.txt`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .selector import APCSSelector


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="apcs",
        description="Arabic-Aware Prompt Compression Selector: recommends a "
                    "compression method and keep-rate for an Arabic prompt.")
    ap.add_argument("prompt", nargs="?", help="prompt text (or use --file)")
    ap.add_argument("--file", type=Path, help="read the prompt from a file")
    ap.add_argument("--rules", type=Path, default=None,
                    help="alternative calibrated rules JSON")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="machine-readable output")
    args = ap.parse_args(argv)

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.prompt:
        text = args.prompt
    else:
        ap.error("provide a prompt or --file")

    rec = APCSSelector(rules_path=args.rules).recommend(text)
    if args.as_json:
        print(json.dumps(rec.as_dict(), ensure_ascii=False, indent=2))
    else:
        f = rec.features
        print(f"recommendation : {rec.method}"
              + (f" @ rate {rec.rate}" if rec.method != "none" else ""))
        print(f"rule fired     : {rec.rule}")
        print(f"features       : tokens={f.token_count} words={f.word_count} "
              f"frag={f.fragmentation_ratio} struct={f.structural_complexity}")
        print(f"rules version  : {rec.rules_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
