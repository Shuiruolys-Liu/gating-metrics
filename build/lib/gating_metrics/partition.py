"""gating_metrics.partition — exact method-of-moments (nested ANOVA
cross-product) decomposition of a two-variable covariance across hierarchy
levels (frozen F1 dictionary): between-individual (between-sample),
between-compartment-within-sample, and within-sample residual components.

The partition is exact algebra:
  SSP_tot        = sum_ij (x_ij - xbar)(y_ij - ybar)
  SSP_between    = sum_s n_s (xbar_s - xbar)(ybar_s - ybar)
  SSP_comp       = sum_{s,c} n_sc (xbar_sc - xbar_s)(ybar_sc - ybar_s)
  SSP_within     = SSP_tot - SSP_between - SSP_comp
with bootstrap CIs obtained by resampling INDIVIDUALS (samples), the
independent units of the hierarchy.
"""
import numpy as np
import pandas as pd


def group_stats(df, x="x", y="y", by=("sample", "comp")):
    d = df.copy()
    d["xy"] = d[x] * d[y]
    return d.groupby(list(by)).agg(n=(x, "size"), sx=(x, "sum"), sy=(y, "sum"),
                                   sxy=("xy", "sum")).reset_index()


def partition_from_gs(gs):
    """Partition from per-group sufficient statistics (n, sx, sy, sxy)."""
    N = gs.n.sum()
    xbar, ybar = gs.sx.sum() / N, gs.sy.sum() / N
    SSP_tot = gs.sxy.sum() - N * xbar * ybar
    gs = gs.copy()
    gs["mx"], gs["my"] = gs.sx / gs.n, gs.sy / gs.n
    ss_rows = []
    for s, g in gs.groupby("sample" if "sample" in gs.columns else gs.columns[0]):
        ss_rows.append({"sample": s, "n": g.n.sum(),
                        "mxs": np.average(g.mx, weights=g.n),
                        "mys": np.average(g.my, weights=g.n)})
    ss = pd.DataFrame(ss_rows)
    gcol = "sample" if "sample" in gs.columns else gs.columns[0]
    SSP_B = float((ss.n * (ss.mxs - xbar) * (ss.mys - ybar)).sum())
    merged = gs.merge(ss[[gcol, "mxs", "mys"]], on=gcol)
    if len(gs.columns) > 4:  # compartment level present
        SSP_C = float((merged.n * (merged.mx - merged.mxs) *
                       (merged.my - merged.mys)).sum())
    else:
        SSP_C = 0.0
    SSP_W = SSP_tot - SSP_B - SSP_C
    tot = SSP_tot if SSP_tot != 0 else np.nan
    return dict(n_cells=int(N), n_groups=len(gs), SSP_tot=SSP_tot,
                SSP_between_individual=SSP_B,
                SSP_between_compartment_within=SSP_C, SSP_within=SSP_W,
                share_between_individual=SSP_B / tot if tot == tot else np.nan,
                share_between_compartment=SSP_C / tot if tot == tot else np.nan,
                share_within=SSP_W / tot if tot == tot else np.nan)


def partition(df, x="x", y="y", by=("sample", "comp")):
    """Convenience wrapper: partition + algebraic identity assert."""
    gs = group_stats(df, x, y, by)
    out = partition_from_gs(gs)
    if by == ("sample",):
        assert abs(out["SSP_tot"] - out["SSP_between_individual"]
                   - out["SSP_within"]) < 1e-6 * max(1, abs(out["SSP_tot"]))
    else:
        assert abs(out["SSP_tot"] - out["SSP_between_individual"]
                   - out["SSP_between_compartment_within"]
                   - out["SSP_within"]) < 1e-6 * max(1, abs(out["SSP_tot"]))
    return out


def bootstrap_share_between(df, x="x", y="y", by=("sample", "comp"),
                            n_boot=1000, seed=42):
    """Bootstrap CI of share_between_individual by resampling samples."""
    rng = np.random.default_rng(seed)
    gs = group_stats(df, x, y, by)
    samples = gs["sample"].unique()
    shares = []
    for _ in range(n_boot):
        draw = rng.choice(samples, len(samples), replace=True)
        rep = pd.DataFrame({"sample": draw}).groupby("sample").size().rename("mult").reset_index()
        g2 = gs.merge(rep, on="sample")
        for c in ["n", "sx", "sy", "sxy"]:
            g2[c] = g2[c] * g2.mult
        shares.append(partition_from_gs(g2)["share_between_individual"])
    lo, hi = np.nanpercentile(shares, [2.5, 97.5])
    return float(lo), float(hi)
