"""Exp 10 — run the post-API analysis chain in order, stopping at the first
failure. Every step's output is appended to <outdir>/chain_log.txt.

  python run_chain.py                         # real run on results/
  python run_chain.py --dry-run --responses <partial.jsonl> --outdir <tmp>
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYN = ROOT.parent / "09_synthetic_sensitivity"
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--responses", default=str(ROOT / "results" / "responses.jsonl"))
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--skip-score", action="store_true",
                    help="reuse an existing final_results.csv in --outdir")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dry = ["--dry-run"] if args.dry_run else []
    syn_out = out / "synthetic" if args.dry_run else SYN / "results"
    results_csv = str(out / "final_results.csv")

    steps = [
        ("score + tau + RQ1", [PY, "score_and_analyze.py", "--responses",
                               args.responses, "--outdir", str(out)] + dry),
        ("RQ2 features", [PY, "rq2_final.py", "--indir", str(out)]),
        ("cost analysis", [PY, "cost_analysis.py", "--indir", str(out)]),
        ("APCS final + one-shot test", [PY, "apcs_final.py", "--indir",
                                        str(out)] + dry),
        ("09a synthetic-share sweep", [PY, str(SYN / "sweep_09a.py"),
                                       "--results", results_csv, "--outdir",
                                       str(syn_out)]),
        ("09b synthetic vs sourced", [PY, str(SYN / "analyze_09b.py"),
                                      "--results", results_csv, "--outdir",
                                      str(syn_out)]),
        ("figures + tables", [PY, "make_figures.py", "--indir", str(out),
                              "--syndir", str(syn_out)]),
    ]
    if args.skip_score:
        steps = steps[1:]

    log = (out / "chain_log.txt").open("a", encoding="utf-8")
    for name, cmd in steps:
        t0 = time.time()
        print(f"== {name} ...", flush=True)
        log.write(f"\n===== {name} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        log.write(p.stdout + "\n" + p.stderr)
        log.flush()
        if p.returncode != 0:
            print(f"!! {name} FAILED (exit {p.returncode}) — see chain_log.txt")
            print(p.stderr[-2500:])
            sys.exit(1)
        tail = [l for l in p.stdout.strip().splitlines() if l.strip()][-3:]
        print(f"   ok in {time.time() - t0:.0f}s | " + " | ".join(tail)[:400],
              flush=True)
    print(f"chain complete -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
