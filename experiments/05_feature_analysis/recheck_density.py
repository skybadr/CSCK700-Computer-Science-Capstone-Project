"""Exp 05 erratum check (2026-09-29): recompute morphological density with the
corrected definition (segments per Arabic-letter token; Farasa punctuation and
numeral tokens excluded) and re-run the RQ2 correlations that used it.

The original compute_features.py divided all Farasa segments, punctuation
included, by the orthographic word count, which inflated density in
proportion to punctuation use. Output: results/density_recheck.json
"""

import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "10_final_benchmark"))
from features_final import farasa_density  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PILOT = ROOT.parent.parent / "AraPromptBench_dataset.json"


def main():
    prompts = json.loads(PILOT.read_text(encoding="utf-8"))["prompts"]
    dens = farasa_density([unicodedata.normalize("NFC", p["prompt"])
                           for p in prompts])
    new = pd.Series(dens, index=[p["id"] for p in prompts], name="density_fixed")
    feats = pd.read_csv(RESULTS / "features.csv").set_index("prompt_id")
    feats["density_fixed"] = new
    lab = pd.read_csv(RESULTS / "pareto_labels.csv")
    lab["density_fixed"] = lab.prompt_id.map(new)

    out = {"mean_old": round(float(feats.morphological_density.mean()), 4),
           "mean_fixed": round(float(feats.density_fixed.mean()), 4),
           "rho_old_vs_fixed": round(float(stats.spearmanr(
               feats.morphological_density, feats.density_fixed)[0]), 3),
           "rho_old_vs_structure": round(float(stats.spearmanr(
               feats.morphological_density, feats.structural_complexity)[0]), 3),
           "rho_fixed_vs_structure": round(float(stats.spearmanr(
               feats.density_fixed, feats.structural_complexity)[0]), 3),
           "rq2_out_f1": {}}
    for method in ["llmlingua2", "llmlingua_qwen", "random_deletion"]:
        for rate in [0.3, 0.5, 0.7]:
            sub = lab[(lab.method == method) & (lab.target_rate == rate)]
            ro = stats.spearmanr(sub.morphological_density, sub.out_f1_arabert)
            rf = stats.spearmanr(sub.density_fixed, sub.out_f1_arabert)
            out["rq2_out_f1"][f"{method}@{rate}"] = {
                "rho_old": round(float(ro[0]), 3),
                "rho_fixed": round(float(rf[0]), 3),
                "p_fixed": float(rf[1])}
    (RESULTS / "density_recheck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
