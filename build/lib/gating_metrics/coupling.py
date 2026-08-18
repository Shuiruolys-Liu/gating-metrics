"""gating_metrics.coupling — layer-wise coupling metrics for state-gated
co-expression structures (channel-B methodology of the ARDS gating study).

Definitions are copied verbatim from the frozen F1 decomposition dictionary
(manifest F1_variance_decomposition_gating, registered 2026-08-18):
  coupling metric = Spearman correlation between the hub gene (e.g. KAT2A)
  and the module (e.g. mean of the inflammasome genes), evaluated at three
  layers:
    L_cell  — within-sample, within-compartment, per-cell correlation, plus a
              detected-gene-count partial sensitivity (library-complexity
              background);
    L_indiv — between-individual pseudobulk correlation across samples;
    L_bulk  — bulk-assay correlation across samples.
"""
import numpy as np
from scipy import stats


def spearman_np(a, b, min_n=4):
    """Spearman r and p over the jointly-finite values of a and b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan, np.nan
    return stats.spearmanr(a[m], b[m])


def partial_spearman(a, b, c, min_n=8):
    """Rank-partial Spearman of a vs b controlling c (residualised ranks)."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if m.sum() < min_n:
        return np.nan, np.nan
    ra, rb, rc = (stats.rankdata(v[m]) for v in (a, b, c))

    def resid(y):
        X = np.column_stack([np.ones_like(rc), rc])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta

    ea, eb = resid(ra), resid(rb)
    if np.std(ea) == 0 or np.std(eb) == 0:
        return np.nan, np.nan
    return stats.pearsonr(ea, eb)


def layer_cell(df, group_cols, x="x", y="y", background="ng", min_cells=30):
    """L_cell: per-group within-sample Spearman (x vs y) and its partial
    (controlling `background`, e.g. per-cell detected-gene count).

    df: long dataframe with one row per CELL plus the group columns
    (e.g. ['sample','compartment']); returns per-group r/p plus the
    median-across-groups summary.
    """
    rows = []
    for key, g in df.groupby(group_cols):
        if len(g) < min_cells:
            continue
        r, p = spearman_np(g[x].values, g[y].values)
        rp, pp = partial_spearman(g[x].values, g[y].values, g[background].values) \
            if background in g else (np.nan, np.nan)
        rows.append(dict(zip(group_cols, key if isinstance(key, tuple) else (key,)),
                         n_cells=len(g), r=r, p=p, r_partial=r, p_partial=pp))
    per = pd.DataFrame(rows) if rows else None
    import pandas as pd
    per = pd.DataFrame(rows)
    if len(per):
        summary = dict(median_r=float(np.nanmedian(per.r)),
                       median_r_partial=float(np.nanmedian(per.r_partial)),
                       n_groups=int(len(per)))
    else:
        summary = dict(median_r=np.nan, median_r_partial=np.nan, n_groups=0)
    return per, summary


def layer_indiv(pb, x="x", y="y", stratum=None, min_n=8):
    """L_indiv: between-individual Spearman across sample-level pseudobulk
    values (pb: one row per sample[ x stratum])."""
    if stratum:
        out = {}
        for s, g in pb.dropna(subset=[x, y]).groupby(stratum):
            out[s] = spearman_np(g[x].values, g[y].values) if len(g) >= min_n else (np.nan, np.nan)
        return out
    return spearman_np(pb[x].values, pb[y].values)
