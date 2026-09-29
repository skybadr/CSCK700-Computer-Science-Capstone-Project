# Experiment 09 — Findings: effect of synthetic data (advisor request)

**Run:** 2026-09-30, inside the Exp 10 chain (gpt-5.6-luna, τ = 0.65,
AraBERT). Synthetic probe prompts were measured in the same run as the real
prompts. Dev split only; the test set was never read.

## 09a — Synthetic share of the calibration data: 15% / 30% / 50%

800-prompt dev mixes (instruction 280 / QA 200 / summarisation 200 /
creative 120, fixed length-band quotas), 200 bootstrap repeats per share,
APCS recalibrated on every mix.

| Synthetic share | "Don't compress below" T1 (median, IQR) | Rates chosen | Rule accuracy | "No compression" labels |
|---|---|---|---|---|
| 15% | 40 tokens (20) | 0.5 / 0.3 | 0.380 | 12.6% |
| 30% | 60 tokens (20) | 0.5 / 0.3 | 0.377 | 12.4% |
| 50% | 60 tokens (20) | 0.5 / 0.3 | 0.374 | 12.2% |

Differences across shares are statistically detectable with 200 repeats
(Kruskal–Wallis: T1 p = 0.007, accuracy p = 2e−9, label share p = 2e−8) but
**practically negligible**: the threshold moves by one grid step, the chosen
compression rates never change, accuracy moves by 0.6 points and the label
mix by 0.4 points. Mean prompt length stays 213–216 tokens in every
condition, confirming the length control held.

**Conclusion:** using up to 50% synthetic prompts to calibrate APCS does not
materially change the rule it learns.

## 09b — Do synthetic prompts behave like real ones under compression?

Synthetic (probe) vs real (v2 dev) prompts within the same task type,
LLMLingua-2, output F1, difference weighted to the real prompts' length mix
(bootstrap 95% CI):

| Category | Rate 0.3 | Rate 0.5 | Rate 0.7 |
|---|---|---|---|
| instruction | −0.025 [−0.043, −0.008] | −0.024 [−0.043, −0.005] | −0.022 [−0.042, −0.002] |
| QA | **+0.067** [0.049, 0.085] | +0.006 [−0.016, 0.029] | −0.048 [−0.072, −0.024] |
| summarisation | −0.037 [−0.052, −0.023] | −0.050 [−0.067, −0.034] | −0.029 [−0.046, −0.014] |

8 of 9 intervals exclude zero. Synthetic instruction and summarisation
prompts lose slightly more fidelity under compression (−0.02 to −0.05); QA
flips sign with rate. Synthetic prompts are also compressed marginally
harder (achieved keep 1–2 points lower at the same target), which accounts
for part of the gap. The differences are the same order of magnitude as the
LLMLingua-2 vs random-deletion effect (~0.04), so they are not trivial for
method comparisons.

**Conclusion:** synthetic data is safe for *calibrating the selector* (09a)
but is **not a drop-in substitute for real prompts when measuring
compression fidelity** (09b): it shifts fidelity by amounts comparable to
the method differences being measured. For the dissertation: the benchmark's
headline method comparisons rest on the 850 real corpus prompts plus the 150
synthetic creative prompts, and category-level results for creative should
carry this caveat.
