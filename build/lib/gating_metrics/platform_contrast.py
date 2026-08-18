"""gating_metrics.platform_contrast — same-compartment cross-platform coupling
contrast (the H3 natural-experiment test) and a measurement-condition
compatibility report builder (frozen F1/F2 dictionary).
"""
import numpy as np
from scipy import stats
import pandas as pd


def fisher_z_contrast(r1, n1, r2, n2):
    """Fisher-Z test of two independent correlations (e.g. bulk-array vs
    scRNA-pseudobulk coupling measured in the same biological compartment)."""
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(fisher_z=float(z), p=float(p), sign_reversal=bool(r1 * r2 < 0))


def compatibility_report(rows):
    """Build the measurement-conditions range table (F2).

    rows: list of dicts with keys dataset, platform, compartment, state_class,
          stratum, r, p, n. Classification rule (frozen): DOMINANT-NEGATIVE
          r<=-0.2 & p<0.05; DOMINANT-POSITIVE r>=+0.2 & p<0.05; else
          INDETERMINATE.
    """
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    def classify(r, p):
        if not np.isfinite(r) or not np.isfinite(p):
            return "NON-EVALUABLE"
        if r <= -0.2 and p < 0.05:
            return "DOMINANT-NEGATIVE"
        if r >= 0.2 and p < 0.05:
            return "DOMINANT-POSITIVE"
        return "INDETERMINATE"
    df["classification"] = [classify(r, p) for r, p in zip(df.r, df.p)]
    return df


def atlas_verdict(report, condition_filter):
    """Frozen F2 atlas-level verdict: the individual-level axis principle is
    supported at atlas level iff DOMINANT-NEGATIVE holds in >=80% of evaluable
    cohorts of the given condition class (e.g. acute-inflammation blood)."""
    sub = report[report.state_class.isin(condition_filter)]
    sub = sub[sub.classification != "NON-EVALUABLE"]
    if len(sub) == 0:
        return dict(k=0, n=0, fraction=np.nan, supported=False)
    k = int((sub.classification == "DOMINANT-NEGATIVE").sum())
    frac = k / len(sub)
    return dict(k=k, n=int(len(sub)), fraction=frac, supported=bool(frac >= 0.8))
