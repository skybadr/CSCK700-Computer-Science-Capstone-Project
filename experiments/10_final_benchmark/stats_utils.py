"""Shared statistics for the Exp 10 analysis chain (bootstrap CIs, McNemar)."""

import numpy as np
from scipy.stats import binomtest

N_BOOT = 10_000
SEED = 42


def bootstrap_ci(values, stat=np.mean, n=N_BOOT, seed=SEED, alpha=0.05):
    """Percentile CI for stat(values), resampling units with replacement."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    boots = np.apply_along_axis(stat, 1, v[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)


def paired_bootstrap_ci(a, b, n=N_BOOT, seed=SEED, alpha=0.05):
    """CI for mean(a - b) with a, b paired per unit (same prompts)."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float),
                        n=n, seed=seed, alpha=alpha)


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar on paired correctness vectors -> (a_only, b_only, p)."""
    ca, cb = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    a_only = int(np.sum(ca & ~cb))
    b_only = int(np.sum(~ca & cb))
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return a_only, b_only, float(p)


def fmt_ci(est, lo, hi, digits=3):
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
