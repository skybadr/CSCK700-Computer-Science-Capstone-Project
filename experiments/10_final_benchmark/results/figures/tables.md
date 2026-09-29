# Exp 10 — thesis tables (auto-generated)

## Table A — Policies on the held-out test set (n = 200, τ = 0.65, AraBERT)

| Policy | Accuracy | Acc. (extended labels) | Mean TCR | Mean out-F1 | Below τ | Cost saving vs none | Pareto hit |
|---|---|---|---|---|---|---|---|
| APCS | 0.310 | 0.310 | 0.429 | 0.750 | 39.0% | -34.2% | 0.795 |
| always noop@1.0 | 0.220 | 0.155 | 0.000 | 1.000 | 0.0% | 0.0% | 0.885 |
| always llmlingua2@0.7 | 0.205 | 0.180 | 0.304 | 0.752 | 22.5% | -9.6% | 0.640 |
| always llmlingua2@0.5 | 0.285 | 0.275 | 0.510 | 0.676 | 45.0% | -19.0% | 0.665 |
| always llmlingua2@0.3 | 0.290 | 0.290 | 0.716 | 0.595 | 71.0% | -37.8% | 0.865 |
| always llmlingua_qwen@0.7 | 0.000 | 0.010 | 0.048 | 0.926 | 1.5% | 1.4% | 0.710 |
| always llmlingua_qwen@0.5 | 0.000 | 0.030 | 0.080 | 0.904 | 5.0% | 3.4% | 0.715 |
| always llmlingua_qwen@0.3 | 0.000 | 0.060 | 0.113 | 0.883 | 9.5% | 1.2% | 0.780 |
| random selection | 0.270 | 0.230 | 0.392 | 0.751 | 35.5% | -14.3% | 0.755 |

APCS accuracy 95% CI: [0.245, 0.375]. Rule: {'T1': 80, 'T2': 250, 'r_mid': 0.5, 'r_long': 0.3}.

## Table B — APCS vs each baseline (paired)

| Baseline | Label set | Accuracy diff | 95% CI | McNemar p |
|---|---|---|---|---|
| always noop@1.0 | primary | +0.090 | [0.01, 0.17] | 0.038 |
| always llmlingua2@0.7 | primary | +0.105 | [0.005, 0.205] | 0.048 |
| always llmlingua2@0.5 | primary | +0.025 | [-0.055, 0.105] | 0.61 |
| always llmlingua2@0.3 | primary | +0.020 | [-0.075, 0.115] | 0.75 |
| always llmlingua_qwen@0.7 | extended | +0.300 | [0.235, 0.37] | 2.3e-16 |
| always llmlingua_qwen@0.5 | extended | +0.280 | [0.21, 0.35] | 8.2e-13 |
| always llmlingua_qwen@0.3 | extended | +0.250 | [0.175, 0.33] | 2.9e-09 |
| random selection | primary | +0.040 | [-0.05, 0.135] | 0.45 |

Always-LLMLingua baselines are compared on the extended label set (which admits LLMLingua candidates); under the primary labels they could never be correct.


mBERT sensitivity: τ = 0.75, rule {'T1': 60, 'T2': 400, 'r_mid': 0.5, 'r_long': 0.3}, test accuracy 0.340, beats all baselines: True.

## Table C — RQ1 method comparison (dev, output-level)

| Comparison | n | A | B | Diff | 95% CI | Wilcoxon p |
|---|---|---|---|---|---|---|
| rq1_output_level: llmlingua2_vs_random@0.3 | 800 | 0.594 | 0.554 | +0.040 | [0.0323, 0.047] | 6e-25 |
| rq1_output_level: llmlingua2_vs_random@0.3_mbert | 800 | 0.709 | 0.692 | +0.017 | [0.0124, 0.0213] | 7.8e-15 |
| rq1_output_level: llmlingua2_vs_random@0.3_within_window | 774 | 0.595 | 0.554 | +0.041 | [0.033, 0.0481] | 2.2e-25 |
| rq1_output_level: llmlingua2_vs_random@0.5 | 800 | 0.684 | 0.646 | +0.038 | [0.0292, 0.0461] | 6.3e-20 |
| rq1_output_level: llmlingua2_vs_random@0.5_mbert | 800 | 0.766 | 0.745 | +0.022 | [0.0163, 0.0273] | 2.5e-17 |
| rq1_output_level: llmlingua2_vs_random@0.5_within_window | 769 | 0.686 | 0.647 | +0.039 | [0.0302, 0.0476] | 5.7e-20 |
| rq1_output_level: llmlingua2_vs_random@0.7 | 800 | 0.766 | 0.716 | +0.050 | [0.0411, 0.0591] | 1.6e-29 |
| rq1_output_level: llmlingua2_vs_random@0.7_mbert | 800 | 0.824 | 0.791 | +0.033 | [0.0264, 0.0388] | 2e-27 |
| rq1_output_level: llmlingua2_vs_random@0.7_within_window | 772 | 0.769 | 0.718 | +0.052 | [0.0431, 0.0612] | 2.4e-29 |
| rq1_prompt_level: llmlingua2_vs_random@0.3 | 800 | 0.571 | 0.586 | -0.014 | [-0.0181, -0.0104] | 6.6e-15 |
| rq1_prompt_level: llmlingua2_vs_random@0.5 | 800 | 0.676 | 0.703 | -0.026 | [-0.0296, -0.0227] | 2.4e-44 |
| rq1_prompt_level: llmlingua2_vs_random@0.7 | 800 | 0.787 | 0.815 | -0.028 | [-0.0311, -0.0247] | 1.9e-53 |
| llmlingua2_vs_llmlingua1_compression_matched: @0.3 | 244 | 0.716 | 0.686 | +0.029 | [0.0136, 0.0453] | 1.8e-06 |
| llmlingua2_vs_llmlingua1_compression_matched: @0.5 | 243 | 0.766 | 0.726 | +0.041 | [0.0254, 0.0556] | 1.3e-12 |
| llmlingua2_vs_llmlingua1_compression_matched: @0.7 | 237 | 0.787 | 0.786 | +0.001 | [-0.0121, 0.0147] | 0.18 |
| llmlingua2_vs_llmlingua1_same_target_rate: @0.3 | 244 | 0.630 | 0.686 | -0.056 | [-0.0737, -0.0394] | 1.9e-09 |
| llmlingua2_vs_llmlingua1_same_target_rate: @0.5 | 243 | 0.710 | 0.726 | -0.016 | [-0.0313, -0.0007] | 0.22 |
| llmlingua2_vs_llmlingua1_same_target_rate: @0.7 | 237 | 0.787 | 0.786 | +0.001 | [-0.0121, 0.0147] | 0.18 |