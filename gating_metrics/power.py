"""gating_metrics.power — closed-form power/sample-size for detecting a
reference-stratum Spearman coupling (the gating-metric planning tool).
"""
import numpy as np
from scipy import stats


def n_for_coupling(r_target, alpha=0.05, power=0.8, two_sided=True):
    """Minimum n to detect a true Spearman r at Fisher-Z (normal approx)."""
    z_a = stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z_b = stats.norm.ppf(power)
    zr = np.arctanh(r_target)
    return int(np.ceil((z_a + z_b) ** 2 / zr ** 2 + 3))


def power_at_n(r_true, n, alpha=0.05, two_sided=True):
    """Power of the Fisher-Z test for a true correlation r at sample size n."""
    z_a = stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    se = 1 / np.sqrt(n - 3)
    zr = np.arctanh(r_true)
    return float(stats.norm.cdf(abs(zr) / se - z_a))


def mde_at_n(n, alpha=0.05, power=0.8, two_sided=True):
    """Minimum detectable |r| at given n."""
    z_a = stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z_b = stats.norm.ppf(power)
    r = np.tanh((z_a + z_b) / np.sqrt(n - 3))
    return float(r)
